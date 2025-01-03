import torch
import torch.nn as nn
import math
import copy
import torch as F
import context 
from torch.nn.parameter import Parameter
from collections import namedtuple
import collections.abc as container_abcs
from typing import List, Dict, Any, Tuple, Union, Optional
from easydict import EasyDict
from torch.autograd import Variable, Function
from ding.utils import POLICY_REGISTRY, MODEL_REGISTRY, SequenceType, squeeze
from ding.policy.base_policy import Policy
from ding.model import model_wrap
from ding.torch_utils import Adam, get_lstm, to_device
from ding.model.common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, MultiHead, RainbowHead, \
    QuantileHead, QRDQNHead, DistributionHead, RegressionHead
from ding.model.template.q_learning import parallel_wrapper
from ding.policy.command_mode_policy_instance import EpsCommandModePolicy
from torch.optim.lr_scheduler import ExponentialLR
from ding.utils.data import timestep_collate, default_collate, default_decollate
from ding.rl_utils import v_nstep_td_data, v_nstep_td_error
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, q_nstep_td_error_with_rescale, get_nstep_return_data
from train.policy.utils import get_train_sample_spine
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def hard_sigm(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0.05, max=0.95)
    return output

# probability-based trigger gate
class Trigger_Eval(Function):
    threshold = 0.5
    
    @staticmethod
    def forward(ctx, x):
        # forward : x -> output
        # self.save_for_backward(x)
        output = x > Trigger_Eval.threshold
        return output.float()

    @staticmethod
    def backward(ctx, output_grad):
        # backward: output_grad -> x_grad
        # x = self.saved_tensors
        x_grad = None

        if ctx.needs_input_grad[0]:
            x_grad = output_grad.clone()

        return x_grad
    
class Trigger_Train(Function):
    @staticmethod
    def forward(ctx, x):
        # forward : x -> output
        # self.save_for_backward(x)
        cond = torch.rand(x.shape)
        output = cond < x
        # print("triggre prob:", x, "output:", output)
        
        return output.float()

    @staticmethod
    def backward(ctx, output_grad):
        # backward: output_grad -> x_grad
        # x = self.saved_tensors
        x_grad = None

        if ctx.needs_input_grad[0]:
            x_grad = output_grad.clone()

        return 0.1 * x_grad


class HM_LSTMCell(nn.Module):
    def __init__(self, bottom_size, hidden_size, top_size, slope, last_layer, z_prob = True):
        super(HM_LSTMCell, self).__init__()
        self.bottom_size = bottom_size
        self.hidden_size = hidden_size
        self.top_size = top_size
        self.slope = slope
        self.last_layer = last_layer
        self.z_prob = z_prob
        '''
        U_11 means the state transition parameters from layer l (current layer) to layer l
        U_21 means the state transition parameters from layer l+1 (top layer) to layer l
        W_01 means the state transition parameters from layer l-1 (bottom layer) to layer l
        '''
        self.U_11 = Parameter(torch.zeros((self.bottom_size, 4 * self.hidden_size + 1), dtype = torch.float32),)
        if not self.last_layer:
            self.U_21 = Parameter(torch.zeros((self.top_size, 4 * self.hidden_size + 1), dtype = torch.float32))
        self.W_01 = Parameter(torch.zeros((self.hidden_size, 4 * self.hidden_size + 1), dtype = torch.float32))
        self.bias = Parameter(torch.zeros((4 * self.hidden_size + 1), dtype = torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(self, c, h_bottom, h, h_top, z, z_bottom):
        # h_bottom.size = batch_size * bottom_size
        s_recur = torch.mm(h, self.W_01)
        # s_recur = (1 - z.expand_as(s_recur)) * s_recur
        if not self.last_layer:
            s_topdown_ = torch.mm(h_top, self.U_21)
            s_topdown = z.expand_as(s_topdown_) * s_topdown_
        else:
            s_topdown = Variable(torch.zeros(s_recur.size()), requires_grad=False)
        s_bottomup_ = torch.mm(h_bottom, self.U_11)
        s_bottomup = z_bottom.expand_as(s_bottomup_) * s_bottomup_
        f_s = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(0).expand_as(s_recur)
        # f_s.size = (4 * hidden_size + 1) * batch_size
        f = torch.sigmoid(f_s[:, 0:self.hidden_size])  # hidden_size * batch_size
        i = torch.sigmoid(f_s[:, self.hidden_size:self.hidden_size*2])
        o = torch.sigmoid(f_s[:, self.hidden_size*2:self.hidden_size*3])
        g = torch.tanh(f_s[:, self.hidden_size*3:self.hidden_size*4])
        z_hat = hard_sigm(self.slope, f_s[:, self.hidden_size*4:self.hidden_size*4+1])

        one = Variable(torch.ones(f.size()), requires_grad=False)
        z = z.expand_as(f)
        z_bottom = z_bottom.expand_as(f)
        #FLUSH, COPY and UPDATE
        c_new = (one - z_bottom) * c + z_bottom * (f * c + i * g)
        h_new = o * F.tanh(c_new)

        # print("checking z_hat:", z_hat)
        if self.z_prob:
            z_new = Trigger_Train.apply(z_hat)
        else:
            z_new = Trigger_Eval.apply(z_hat)
        return h_new, c_new, z_new,f.mean(),i.mean(),o.mean()  # (hidden_size, batch_size)


class HM_LSTM_Actor(nn.Module):
    def __init__(self, 
                obs_shape: Union[int, SequenceType],
                action_shape: Union[int, SequenceType],
                slope = 0.1, 
                size_list: list = [64, 64],
                encoder_hidden_size_list: SequenceType = [64, 64],
                activation: Optional[nn.Module] = nn.ReLU(),
                norm_type: Optional[str] = None,
                head_layer_num: int = 1,
                single_layer = False):
        super(HM_LSTM_Actor, self).__init__()
        self.slope = slope
        self.input_size = obs_shape
        self.size_list = size_list
        self.encoder_hidden_size_list = encoder_hidden_size_list
        self.single_layer = single_layer
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)

        self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        if self.single_layer:
            self.cell_1 = HM_LSTMCell(self.encoder_hidden_size_list[-1], self.size_list[0], None, self.slope, True, z_prob = True)
            self.head = RegressionHead(
                self.size_list[0], action_shape, head_layer_num, 
                final_tanh = True, 
                activation=activation, 
                norm_type=norm_type
            )
        else:
            self.cell_1 = HM_LSTMCell(self.encoder_hidden_size_list[-1], self.size_list[0], self.size_list[1], self.slope, False, z_prob = True)
            self.cell_2 = HM_LSTMCell(self.size_list[0], self.size_list[1], None, self.slope, True, z_prob = True)
            self.head = RegressionHead(
                    self.size_list[1], action_shape, head_layer_num, 
                    final_tanh = True, 
                    activation=activation, 
                    norm_type=norm_type
                )
        
    def set_slope(self, slope):
        self.cell_1.slope = slope
        # self.cell_2.slope = slope
        
    def get_slope(self):
        return self.cell_1.slope
        
        
    def forward(self, x: Dict, prev_state: tuple,
                inference: bool = False, burning: bool = False,
                saved_hidden_state_timesteps: Optional[list] = None):
        # inputs.size = ( time steps, batch_size, embed_size/input_size)
        if inference:
            time_steps = 1
            batch_size = x.size(0)
        else:
            time_steps = x.size(0)
            batch_size = x.size(1)
        if prev_state == None:
            if self.single_layer:
                h_t1 = Variable(torch.zeros(batch_size, self.size_list[0]).float(), requires_grad=False)
                c_t1 = Variable(torch.zeros(batch_size, self.size_list[0]).float(), requires_grad=False)
                z_t1 = Variable(torch.ones(batch_size, 1).float(), requires_grad=False)
                
            else: 
                h_t1 = Variable(torch.zeros(batch_size, self.size_list[0]).float(), requires_grad=False)
                c_t1 = Variable(torch.zeros(batch_size, self.size_list[0]).float(), requires_grad=False)
                z_t1 = Variable(torch.ones(batch_size, 1).float(), requires_grad=False)
                h_t2 = Variable(torch.zeros(batch_size, self.size_list[1]).float(), requires_grad=False)
                c_t2 = Variable(torch.zeros(batch_size, self.size_list[1]).float(), requires_grad=False)
                z_t2 = Variable(torch.zeros(batch_size, 1).float(), requires_grad=False)
        else:
            if self.single_layer:
                (h_t1, c_t1, z_t1) = prev_state
            else:
                (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = prev_state
        z_one = Variable(torch.ones(batch_size, 1, ).float(), requires_grad=False)
        z_zeros = Variable(torch.zeros(batch_size, 1, ).float(), requires_grad=False)
        lstm_embedding = []
        # for both inference and other cases, the network structure is encoder -> rnn network -> head
        # the difference is inference take the data with seq_len=1 (or T = 1)
        if inference:
            
            x = self.encoder(x)
            if self.single_layer:
                # print("prev_state in actor:", (h_t1, c_t1, z_t1))
                h_t1, c_t1, z_t1, _, _, _ = self.cell_1(c=c_t1, h_bottom=x, h=h_t1, h_top=None, z=z_zeros, z_bottom=z_one)
                x = self.head(h_t1)
                x['next_state'] = (h_t1, c_t1, z_t1)
                x['trigger'] = z_one
            else:
                h_t1, c_t1, z_t1, _, _, _ = self.cell_1(c=c_t1, h_bottom=x, h=h_t1, h_top=h_t2, z=z_t1, z_bottom=z_one)
                if z_t1.item() != 0:
                    h_t2, c_t2, z_t2, _, _, _ = self.cell_2(c=c_t2, h_bottom=h_t1, h=h_t2, h_top=None, z=z_zeros, z_bottom=z_t1)
                x = self.head(h_t2)
                x['next_state'] = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
                x['trigger'] = z_t1
            return x
        else:
            assert len(x.shape) in [3, 5], x.shape
            x = parallel_wrapper(self.encoder)(x) # (T,B,N)
            if saved_hidden_state_timesteps is not None:
                saved_hidden_state = []
            z = []
            h = []
            f_1_means = []
            f_2_means = []
            i_1_means = []
            i_2_means = []
            o_1_means = []
            o_2_means = []
            for t in range(x.shape[0]):  # T timesteps
                
                if self.single_layer:
                    h_t1, c_t1, z_t1, f_mean, i_mean, o_mean = self.cell_1(c=c_t1, h_bottom=x[t, :, :], h=h_t1, h_top=None, z=z_zeros, z_bottom=z_one)
                    f_1_means.append(f_mean)
                    i_1_means.append(i_mean)
                    o_1_means.append(o_mean)
                    next_state = (h_t1, c_t1, z_t1)
                    lstm_embedding.append(h_t1) # h_t2: (B, N)
                    z.append(z_one)
                else:
                    h_t1, c_t1, z_t1, f_mean, i_mean, o_mean = self.cell_1(c=c_t1, h_bottom=x[t, :, :], h=h_t1, h_top=h_t2, z=z_t1, z_bottom=z_one)
                    h_t2, c_t2, z_t2, f_mean_2, i_mean_2, o_mean_2 = self.cell_2(c=c_t2, h_bottom=h_t1, h=h_t2, h_top=None, z=z_zeros, z_bottom=z_t1)  # 0.01s used
                    f_1_means.append(f_mean)
                    f_2_means.append(f_mean_2)
                    i_1_means.append(i_mean)
                    i_2_means.append(i_mean_2)
                    o_1_means.append(o_mean)
                    o_2_means.append(o_mean_2)
                    next_state = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
                    lstm_embedding.append(h_t2) # h_t2: (B, N)
                    z.append(z_t1)
                if saved_hidden_state_timesteps is not None and t + 1 in saved_hidden_state_timesteps:
                    saved_hidden_state.append(next_state)
                # TODO: 怎么修改的？
                # hidden_state = list(zip(*prev_state))  # [(h,z,c...) for batch1, (h,z,c...) for batch 2...]
                # only keep ht, {list: x.shape[0]{Tensor:(1, batch_size, head_hidden_size)}}
                # hidden_state_list.append(torch.cat(hidden_state[0], dim=1))
                
            if not burning:
                x = torch.stack(lstm_embedding, 0)  # (T, B, head_hidden_size)
                # x = h_t2.unsqueeze(0)
                x = parallel_wrapper(self.head)(x) # (T, B, action_shape)
                if len(x['pred'].shape) == 2:
                    x['pred'] = x['pred'].unsqueeze(2)
                x['trigger'] = torch.stack(z, dim = 0)
            else:
                x = dict()  
            f_1_mean = torch.stack(f_1_means).mean()
            i_1_mean = torch.stack(i_1_means).mean()
            o_1_mean = torch.stack(o_1_means).mean()
            if self.single_layer:
                f_2_mean, i_2_mean, o_2_mean = 0., 0., 0.
            else:
                f_2_mean = torch.stack(f_2_means).mean()
                i_2_mean = torch.stack(i_2_means).mean()
                o_2_mean = torch.stack(o_2_means).mean()
                
            x['next_state'] = next_state 
            x['gates'] = [f_1_mean, f_2_mean,i_1_mean, i_2_mean,o_1_mean, o_2_mean]
            # x['hidden_state'] = 
            if saved_hidden_state_timesteps is not None:
                x['saved_hidden_state'] = saved_hidden_state  # the selected saved hidden states, including h and c
            return x


class HM_LSTM_Critic(nn.Module):
    def __init__(self, 
                obs_shape: Union[int, SequenceType], #obs_shape = obs_shape + watcher action 
                action_shape: Union[int, SequenceType],
                slope = 1, 
                size_list: list = [64, 64],
                encoder_hidden_size_list: SequenceType = [64, 64],
                activation: Optional[nn.Module] = nn.ReLU(),
                norm_type: Optional[str] = None,
                head_layer_num: int = 1):
        super(HM_LSTM_Critic, self).__init__()
        self.slope = slope
        self.input_size = obs_shape
        self.size_list = size_list
        self.encoder_hidden_size_list = encoder_hidden_size_list
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)

        self.encoder = FCEncoder(self.input_size, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        self.cell_1 = HM_LSTMCell(self.encoder_hidden_size_list[-1], self.size_list[0], None, self.slope, True, z_prob = False)
        # self.cell_2 = HM_LSTMCell(self.size_list[0], self.size_list[1], None, self.slope, True, z_prob = False)
        self.head = RegressionHead(
                self.size_list[0] + action_shape, 1, head_layer_num, 
                final_tanh = False, 
                activation=activation, 
                norm_type=norm_type
            )
    
    def set_slope(self, slope):
        self.cell_1.slope = slope
        # self.cell_2.slope = slope
        
    def get_slope(self):
        return self.cell_1.slope
    
    def forward(self, x: torch.Tensor, prev_state: tuple, action, burning: bool = False,
                saved_hidden_state_timesteps: Optional[list] = None):
        # inputs.size = ( time steps, batch_size, embed_size/input_size)

        time_steps = x.size(0)
        batch_size = x.size(1)
        
        if prev_state == None or (len(prev_state) == 1 and prev_state[0] is None):
            h_t1 = Variable(torch.zeros(batch_size, self.size_list[0]).float(), requires_grad=False)
            c_t1 = Variable(torch.zeros(batch_size, self.size_list[0]).float(), requires_grad=False)
            z_t1 = Variable(torch.ones(batch_size, 1, ).float(), requires_grad=False)
            # h_t2 = Variable(torch.zeros(batch_size, self.size_list[1]).float(), requires_grad=False)
            # c_t2 = Variable(torch.zeros(batch_size, self.size_list[1]).float(), requires_grad=False)
            # z_t2 = Variable(torch.zeros(batch_size, 1).float(), requires_grad=False)
        else:
            (h_t1, c_t1, z_t1) = prev_state
            
        z_one = Variable(torch.ones(batch_size, 1 ).float(), requires_grad=False)
        z_zeros = Variable(torch.zeros(batch_size, 1, ).float(), requires_grad=False)
        
            
        # for both inference and other cases, the network structure is encoder -> rnn network -> head
        # the difference is inference take the data with seq_len=1 (or T = 1)
        assert len(x.shape) in [3, 5], x.shape
        # print("x shape in critic:", x.shape)
        x = parallel_wrapper(self.encoder)(x) # (T,B,N)
        if saved_hidden_state_timesteps is not None:
            saved_hidden_state = []
        z = []
        h = []
        f_1_means = []
        i_1_means = []
        o_1_means = []
        lstm_embedding = []
        for t in range(x.shape[0]):  # T timesteps
            
            h_t1, c_t1, z_t1, f_mean, i_mean, o_mean = self.cell_1(c=c_t1, h_bottom=x[t, :, :], h=h_t1, h_top=None, z=z_zeros, z_bottom=z_one)
            # h_t2, c_t2, z_t2 = self.cell_2(c=c_t2, h_bottom=h_t1, h=h_t2, h_top=None, z=z_zeros, z_bottom=z_t1)  # 0.01s used
            next_state = (h_t1, c_t1, z_t1)
            f_1_means.append(f_mean)
            i_1_means.append(i_mean)
            o_1_means.append(o_mean)
            if saved_hidden_state_timesteps is not None and t + 1 in saved_hidden_state_timesteps:
                saved_hidden_state.append(next_state)
            lstm_embedding.append(h_t1) # h_t2: (B, N)
            # TODO: 怎么修改的？
            # hidden_state = list(zip(*prev_state))  # [(h,z,c...) for batch1, (h,z,c...) for batch 2...]
            # only keep ht, {list: x.shape[0]{Tensor:(1, batch_size, head_hidden_size)}}
            # hidden_state_list.append(torch.cat(hidden_state[0], dim=1))
        # lstm_embedding.append(h_t2) # h_t2: (B, N)
        if not burning:
            x = torch.stack(lstm_embedding, 0)  # (T, B, head_hidden_size)
            # print("checking x:", x.shape,
            #       "\n action:", action.shape)
            x = torch.cat([x, action], dim = 2)  # (T, B, head_hidden_size)
            x = parallel_wrapper(self.head)(x) # (T, B, action_shape)
            if len(x['pred'].shape) == 2:
                x['pred'] = x['pred'].unsqueeze(2)
        else:
            x = dict()
        f_1_mean = torch.stack(f_1_means).mean()
        i_1_mean = torch.stack(i_1_means).mean()
        o_1_mean = torch.stack(o_1_means).mean()
        x['next_state'] = next_state 
        x['gates'] = [f_1_mean, i_1_mean, o_1_mean]
        
        # x['hidden_state'] = 
        if saved_hidden_state_timesteps is not None:
            x['saved_hidden_state'] = saved_hidden_state  # the selected saved hidden states, including h and c
        return x

@MODEL_REGISTRY.register('spine_model') 
class SpineModel(nn.Module):
    mode = ['compute_actor', 'compute_critic']
    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            global_obs_shape,
            action_shape: Union[int, SequenceType, EasyDict],
            twin_critic: bool = False,
            size_list: list = [64, 64],
            encoder_hidden_size_list: SequenceType = [128, 64],
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            lstm_type: Optional[str] = 'normal',
            slope = 0.1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            single_layer : bool = False
    ) -> None:
        super(SpineModel, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        # shared encoder&lstm across critic and actor
        self.slope = slope
        self.twin_critic = twin_critic
        self.actor = HM_LSTM_Actor(obs_shape, action_shape, slope, size_list= size_list,
                                   encoder_hidden_size_list= encoder_hidden_size_list,
                                   activation= activation,
                                   norm_type= norm_type,
                                   head_layer_num = head_layer_num,
                                   single_layer = single_layer
                                   )
        
        critic_input_size = global_obs_shape
        
        if self.twin_critic:
            self.critic = nn.ModuleList()
            for _ in range(2):
                self.critic.append(   
                    HM_LSTM_Critic(critic_input_size, 
                        action_shape + 1, 
                        slope, 
                        size_list,
                        encoder_hidden_size_list= encoder_hidden_size_list,
                        activation= activation,
                        norm_type= norm_type,
                        head_layer_num = head_layer_num
                    )
                )
        else:
            self.critic = HM_LSTM_Critic(critic_input_size, 
                        action_shape + 1, 
                        slope, 
                        size_list,
                        encoder_hidden_size_list= encoder_hidden_size_list,
                        activation= activation,
                        norm_type= norm_type,
                        head_layer_num = head_layer_num
                    )
        
    def set_slope(self, slope):
        self.actor.set_slope(slope)
        if self.twin_critic:
            for i in range(len(self.critic)):
                self.critic[i].set_slope(slope) 
        else:
            self.critic.set_slope(slope)
                
    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str, **kwargs) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        # print("model fed inputs:", inputs, "\nmode:", mode)
        return getattr(self, mode)(inputs, **kwargs)

    def compute_actor(self, inputs: Dict, burning: bool = False, **kwargs) -> Dict:
        x = self.actor(inputs['obs']['agent_state'], inputs['prev_state'], 
                       burning = burning, **kwargs)
        if burning:
            return x
        else:
            return {'action': x['pred'],
                        **{key : x[key] for key in x.keys() if key != 'pred'}}

    def compute_critic(self, inputs: Dict, burning: bool = False, **kwargs) -> Dict:
        obs, prev_state = inputs['obs']['global_state'], inputs['prev_state']
        if prev_state == [None]:
            if self.twin_critic:
                prev_state = [None, None]
            else:
                prev_state = None
        if not burning:
            action = inputs['action'] # (T, B ,1 )
            trigger = inputs['trigger'] # (T, B, 1)
            # print(f"what is the action:{action}"
            #        f"and trigger:{trigger}")
            action = torch.cat([action, trigger], dim = 2)
            # print("resulting action:", action)
        else:
            action = None
        # x = torch.cat([obs, action], dim=1)
        if self.twin_critic:
            x = [m(obs, prev_state = prev_state[i], 
                   action = action, burning = burning, **kwargs) for i,m in enumerate(self.critic)]
            if burning:
                return {key : [m[key] for m in x] for key in x[0].keys()}
            else:
                return {'q_value': [m['pred'] for m in x],
                        **{key : [m[key] for m in x] for key in x[0].keys() if key != 'pred'}}
        else:
            x = self.critic(obs, prev_state = prev_state, 
                            action = action, burning = burning, **kwargs)
            if burning:
                return x
            else:
                return {'q_value' : x['pred'],
                        **{key : x[key] for key in x.keys() if key != 'pred'}}


@POLICY_REGISTRY.register('spine_policy')
class SpinePolicy(Policy):
    r"""
        Spine Policy works as follows:
            It consists of an actor and a critic, while the critic use naive LSTM structure,
            the actor consists of watchers, policers and executors:
                - Watchers trigger the policiers.
                - Policers generate new executors.
                - Executors response to signals with actions.
            During the interaction, the actor works in two different time intervals,  
            
    """
    config = dict(
        type = 'spine',
        cuda = False,
        on_policy = False,
        multi_agent = False,
        priority = False,
        priority_IS_weight = False,
        action_shape = 'continuous',
        reward_batch_norm = False,
        # r2d2 related
        burnin_step=2,
        unroll_len = 80,
        nstep = 5,
        
        
        model = dict(
            twin_critic = False,
        ) ,
        discount_factor = 0.99,
        learn = dict(
            multi_gpu = False,
            update_per_collect = 1,
            batch_size = 256,
            learning_rate_actor = 1e-3,
            learning_rate_critic = 1e-3,
            ignore_done = False,
            target_theta = 0.005,
            actor_update_freq = 10,
            noise = False,
        ),
        collect=dict(
            # need n_sample or n_episode
            n_episode = 1,
            noise_sigma = 0.1,
        ),
        eval=dict(
            evaluator = dict(
                eval_freq = 5000,
            )
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
            )
        )
    )
    
    
    def _init_learn(self)->None:
        r"""
            Called by self.__init__
        """
        
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._actor_optimizer = Adam(self._model.actor.parameters(), lr=self._cfg.learn.learning_rate_actor)
        self._critic_optimizer = Adam(self._model.critic.parameters(), lr=self._cfg.learn.learning_rate_critic)
        self._critic_schedular = ExponentialLR(self._critic_optimizer, gamma = self._cfg.learn.learning_rate_gamma)
        self._actor_schedular = ExponentialLR(self._actor_optimizer, gamma = self._cfg.learn.learning_rate_gamma ** self._cfg.learn.actor_update_freq)
        
        self._gamma = self._cfg.discount_factor
        # multiscale 
        self._init_slope =  self._cfg.learn.init_slope
        self._slope_decay =  self._cfg.learn.slope_decay
        self._max_slope = self._cfg.learn.max_slope
        # td3
        self._actor_update_freq = self._cfg.learn.actor_update_freq
        self._twin_critic = self._cfg.model.twin_critic
        # r2d2
        self._burnin_step = self._cfg.burnin_step
        self._nstep = self._cfg.nstep 
        # create target model 
        self._target_model = copy.deepcopy(self._model)
        
        self._model.set_slope(self._init_slope)
        self._target_model.set_slope(self._init_slope)
        
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name= 'finer_target',
            update_type = 'momentum',
            update_kwargs = {'theta': self._cfg.learn.target_theta}
        )
        if self._cfg.learn.noise:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='action_noise',
                noise_type='gauss',
                noise_kwargs={
                    'mu': 0.0,
                    'sigma': self._cfg.learn.noise_sigma
                },
                noise_range=self._cfg.learn.noise_range
            )
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='hidden_state_ac',
            state_num=self._cfg.learn.batch_size,
        )
        
        # create learn model
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state_ac',
            state_num=self._cfg.learn.batch_size,
        )
        
        self._learn_model.reset()
        self._target_model.reset()
        self._forward_learn_cnt = 0
        

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]) -> dict:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least \
                ['main_obs', 'target_obs', 'burnin_obs', 'action', 'reward', 'done', 'weight']
            - data_info (:obj:`dict`): the data info, such as replay_buffer_idx, replay_unique_id
        """
        # data preprocess
        # print("data before timestep collate:", data)
        data = timestep_collate(data)
        for i in range(len(data['prev_state'])):
            data['prev_state'][i] = torch.stack(data['prev_state'][i], dim = 0)
        if data['action'] is not None and len(data['action'].shape) == 2:  # (T, B, ) -> (T, B, 1)
                data['action'] = data['action'].unsqueeze(2)    
        if data['trigger'] is not None and len(data['trigger'].shape) == 2:  # (T, B, ) -> (T, B, 1)
                data['trigger'] = data['trigger'].unsqueeze(2)
        if data['reward'] is not None and len(data['reward'].shape) == 2:  # (T, B, ) -> (T, B, 1)
                data['reward'] = data['reward'].unsqueeze(2)    
        if data['done'] is not None and len(data['done'].shape) == 2:  # (T, B, ) -> (T, B, 1)
                data['done'] = data['done'].unsqueeze(2)    
        # print("data after timestep collate:", data)
        if self._cuda:
            data = to_device(data, self._device)

        if self._priority_IS_weight:
            assert self._priority, "Use IS Weight correction, but Priority is not used."
        if self._priority and self._priority_IS_weight:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', None)

        burnin_step = self._burnin_step

        # data['done'], data['weight'], data['value_gamma'] is used in def _forward_learn() to calculate
        # the q_nstep_td_error, should be length of [self._unroll_len_add_burnin_step-self._burnin_step]
        ignore_done = self._cfg.learn.ignore_done
        if ignore_done:
            data['done'] = [0 for _ in range(self._unroll_len_add_burnin_step - burnin_step)]
        else:
            data['done'] = data['done'][burnin_step:].float()  # for computation of online model self._learn_model
            # NOTE that after the proprocessing of  get_nstep_return_data() in _get_train_sample
            # the data['done'] [t] is already the n-step done

        # if the data don't include 'weight' or 'value_gamma' then fill in None in a list
        # with length of [self._unroll_len_add_burnin_step-self._burnin_step],
        # below is two different implementation ways
        if 'value_gamma' not in data:
            data['value_gamma'] = [None for _ in range(self._unroll_len_add_burnin_step - burnin_step)]
        else:
            data['value_gamma'] = data['value_gamma'][burnin_step:]

        if 'weight' not in data or data['weight'] is None:
            data['weight'] = [None for _ in range(self._unroll_len_add_burnin_step - burnin_step)]
        else:
            data['weight'] = data['weight'] * torch.ones_like(data['done'])
            # every timestep in sequence has same weight, which is the _priority_IS_weight in PER

        data['action'] = data['action'][burnin_step: -self._nstep]
        data['trigger'] = data['trigger'][burnin_step: -self._nstep]
        data['reward'] = data['reward'][burnin_step: -self._nstep]

        # the burnin_nstep_obs is used to calculate the init hidden state of rnn for the calculation of the q_value,
        # target_q_value, and target_q_action

        # 这样main_obs和target_obs的长度相等，即学习的长度相同。
        # these slicing are all done in the outermost layer, which is the seq_len dim
        data['burnin_obs'] = {key: data['obs'][key][:burnin_step] for key in data['obs'].keys()}
        data['burnin_nstep_obs'] = {key: data['obs'][key][:burnin_step + self._nstep] for key in data['obs'].keys()}
        # the main_obs is used to calculate the q_value, the [bs:-self._nstep] means using the data from
        # [bs] timestep to [self._unroll_len_add_burnin_step-self._nstep] timestep
        data['main_obs'] = {key : data['obs'][key][burnin_step:-self._nstep] for key in data['obs'].keys()}
        # the target_obs is used to calculate the target_q_value
        data['target_obs'] = {key: data['obs'][key][burnin_step + self._nstep:] for key in data['obs'].keys()}
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
            Acquire the data, calculate the loss and optimize learner model.

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['main_obs', 'target_obs', 'burnin_obs', 'action', 'reward', 'done', 'weight']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
        """
        # forward
        # print("forward learning!")
        # print("data before preprocessing: ", data)
        data = self._data_preprocess_learn(data)  # output datatype: Dict
        # print("data after preprocessing: ", data)
        self._learn_model.train()
        self._target_model.train()
        
        # print("new slope:", min(self._init_slope + self._slope_decay * self._forward_learn_cnt, self._max_slope ))
        self._model.set_slope(min(self._init_slope + self._slope_decay * self._forward_learn_cnt, self._max_slope ))
        self._target_model.set_slope(min(self._init_slope + self._slope_decay * self._forward_learn_cnt, self._max_slope))
        
        # use the hidden state in timestep=0
        # note the reset method is performed at the hidden state wrapper, to reset self._state.
        
        trigger_mean = data['trigger'].mean()
        # ====================
        # critic learn forward
        # ====================
        burning_inputs = {'obs': data['burnin_obs'], 'enable_fast_timestep': True}
        burning_nstep_inputs = {'obs': data['burnin_nstep_obs'], 'enable_fast_timestep': True}
        inputs = {'obs': data['main_obs'], 'action' : data['action'], 'trigger': data['trigger'], 'enable_fast_timestep': True}
        next_inputs = {'obs': data['target_obs'], 'enable_fast_timestep': True}
        # all_inputs = {'obs': data['obs'], 'enable_fast_timestep': True}
        self._learn_model.reset(actor_state=data['prev_state'])
        self._target_model.reset(actor_state=data['prev_state'])
        if self._burnin_step != 0:
            with torch.no_grad():
                burnin_critic_output = self._learn_model.forward(
                    burning_inputs, mode = "compute_critic", burning = True, #saved_hidden_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep]
                )  
        with torch.no_grad():
            self._target_model.forward(
                burning_nstep_inputs, mode = "compute_critic", burning = True, #saved_hidden_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep]
            )  
            self._target_model.forward(
                burning_nstep_inputs, mode = "compute_actor", burning = True, #saved_hidden_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep]
            )
                # keys include 'logit', 'hidden_state' 'saved_hidden_state', \
                # 'action', for their specific dim, please refer to DRQN model
        critic_data = self._learn_model.forward(inputs, mode = "compute_critic")
        q_value = critic_data['q_value']
        critic_gates = critic_data['gates']
        q_value_dict = {}
        if self._twin_critic:
            q_value_dict['q_value_t0'] = q_value[0][0].mean()
            q_value_dict['q_value_t0_twin'] = q_value[1][0].mean()
        else:
            q_value_dict['q_value_t0'] = q_value[0].mean()
        # target_q_value
        with torch.no_grad():
            next_actor_data = self._target_model.forward(next_inputs, mode = "compute_actor")
            next_actor_data['obs'] = data['target_obs']
            # print("checking next actor data:", next_actor_data)
            target_q_value = self._target_model.forward(next_actor_data, mode = 'compute_critic')['q_value']
        actor_gates = next_actor_data['gates']
        # T, B, nstep -> T, nstep, B
        reward = data['reward'].permute(0, 2, 1).contiguous()
        td_error = []
        loss_dict = {}
        loss = []
        loss_dict['critic_loss'] = []
        if self._twin_critic:
            loss_dict['critic_twin_loss'] = []
    
        # print("checking target q value: ", target_q_value)
        # print("checking the data needed for td error: ", reward, data['done'], data['weight'], data['value_gamma'])
        for t in range(self._unroll_len_add_burnin_step - self._burnin_step - self._nstep - 1, self._unroll_len_add_burnin_step - self._burnin_step - self._nstep):
            # print(f"calculating {t} step.")
            if self._twin_critic:
                # TD3: two critic networks
                target_q_value_min = torch.min(target_q_value[0][t], target_q_value[1][t])  # find min one as target q value
                # critic network1
                td_data = v_nstep_td_data(q_value[0][t], target_q_value_min, reward[t], data['done'][t], data['weight'][t], data['value_gamma'][t])

                critic_loss, td_error_per_sample1 = v_nstep_td_error(td_data, self._gamma, self._nstep)
                loss_dict['critic_loss'].append(critic_loss)
                # critic network2(twin network
                td_data_twin = v_nstep_td_data(q_value[1][t], target_q_value_min, reward[t], data['done'][t], data['weight'][t], data['value_gamma'][t])
                critic_twin_loss, td_error_per_sample2 = v_nstep_td_error(td_data_twin, self._gamma, self._nstep)
                loss_dict['critic_twin_loss'].append(critic_twin_loss)
                td_error_per_sample = (td_error_per_sample1.abs() + td_error_per_sample2.abs()) / 2
            else:
                # DDPG: single critic network
                td_data = v_nstep_td_data(q_value[t], target_q_value[t], reward[t], data['done'][t], data['weight'][t], data['value_gamma'][t])
                # print("the td_data: ", td_data)
                critic_loss, td_error_per_sample = v_nstep_td_error(td_data, self._gamma, self._nstep)
                loss_dict['critic_loss'].append(critic_loss)
            # td_error.append(td_e, rror_per_sample.abs())
        # print("critic loss before:", loss_dict['critic_loss'])
        loss_dict['critic_loss'] = sum(loss_dict['critic_loss']) / (len(loss_dict['critic_loss']) + 1e-8)
        # print("critic loss after:", loss_dict['critic_loss'])
        if self._twin_critic:
            loss_dict['critic_twin_loss'] = sum(loss_dict['critic_twin_loss']) / (len(loss_dict['critic_twin_loss']) + 1e-8)
        # using the mixture of max and mean absolute n-step TD-errors as the priority of the sequence
        # td_error_per_sample = 0.9 * torch.max(
        #     torch.stack(td_error), dim=0
        # )[0] + (1 - 0.9) * (torch.sum(torch.stack(td_error), dim=0) / (len(td_error) + 1e-8))
        # torch.max(torch.stack(td_error), dim=0) will return tuple like thing, please refer to torch.max
        # td_error shape list(<self._unroll_len_add_burnin_step-self._burnin_step-self._nstep>, B), for example, (75,64)
        # torch.sum(torch.stack(td_error), dim=0) can also be replaced with sum(td_error)
        
        self._critic_optimizer.zero_grad()
        if self._twin_critic:
            loss_dict['critic_twin_loss'].backward()
        loss_dict['critic_loss'].backward()
        # print("learn critic network gradients: ")
        # for name, parameter in self._learn_model.critic.named_parameters():
        #     print(f"{name} : {parameter} \n grad : {parameter.grad}")
            
        # print("target critic network gradients: ")
        # for name, parameter in self._target_model.critic.named_parameters():
        #     print(f"{name} : {parameter} \n grad : {parameter.grad}")
            
        self._critic_optimizer.step()
        self._target_model.update_critic(self._learn_model.state_dict())
        self._forward_learn_cnt += 1
        
        # update for actor
        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            # for actor
            self._learn_model.reset(actor_state=data['prev_state'])
            self._target_model.reset(actor_state=data['prev_state'])
            
            if self._burnin_step != 0:
                with torch.no_grad():
                    self._learn_model.forward(
                        burning_inputs, burning = True, mode = "compute_actor"
                    )
                    self._learn_model.forward(
                        burning_inputs, burning = True, mode = "compute_critic"
                    )
            actor_data = self._learn_model.forward(inputs, mode='compute_actor')
            # actor has less action output, thus need some transformation
            actor_data['obs'] = data['main_obs']
            actor_q_value = self._learn_model.forward(actor_data, mode='compute_critic')['q_value']
            # print('actor q value: ', actor_q_value.shape, actor_q_value)
            if self._twin_critic:
                actor_loss = -actor_q_value[0][-1].mean()
            else:
                actor_loss = -actor_q_value[-1].mean()
            loss_dict['actor_loss'] = actor_loss
            # actor update
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_optimizer.step()
            self._actor_schedular.step()
            self._target_model.update_actor(self._learn_model.state_dict())
        self._critic_schedular.step()
        return {
            'cur_lr_actor': self._actor_schedular.get_last_lr()[0],
            'cur_lr_critic': self._critic_schedular.get_last_lr()[0],
            'trigger' : trigger_mean,
            'actor_f_1': actor_gates[0], 
            'actor_f_2': actor_gates[1],
            'actor_i_1': actor_gates[2],
            'actor_i_2': actor_gates[3],
            'actor_o_1': actor_gates[4],
            'actor_o_2': actor_gates[5],
            'critic_f_1': critic_gates[0],
            'critic_i_1': critic_gates[1],
            'critic_o_1': critic_gates[2],
            # 'priority': td_error_per_sample.tolist(), 
            **q_value_dict,
            **loss_dict,
        }
        
        
    def _reset_learn(self) -> None:
        self._learn_model.reset()

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_actor': self._actor_optimizer.state_dict(),
            'optimizer_critic': self._critic_optimizer.state_dict()
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._actor_optimizer.load_state_dict(state_dict['optimizer_actor'])
        self._critic_optimizer.load_state_dict(state_dict['optimizer_critic'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        assert 'unroll_len' not in self._cfg.collect, "r2d2 use default unroll_len"
        
        self._nstep = self._cfg.nstep
        self._burnin_step = self._cfg.burnin_step
        self._gamma = self._cfg.discount_factor
        self._unroll_len_add_burnin_step = self._cfg.unroll_len + self._cfg.burnin_step
        self._unroll_len = self._unroll_len_add_burnin_step  # for compatibility
        self._unroll_overlap = self._cfg.unroll_overlap

        # for r2d2, this hidden_state wrapper is to add the 'prev hidden state' for each transition.
        # Note that collect env forms a batch and the key is added for the batch simultaneously.

        self._collect_model = model_wrap(
            self._model,
            wrapper_name='action_noise',
            noise_type='hybrid',
            noise_kwargs={
                'mu': 0.0,
                'sigma': self._cfg.collect.noise_sigma,
                'noise_exp': self._cfg.collect.noise_exp,
                'noise_end': self._cfg.collect.noise_end,
                'random_exp': self._cfg.collect.random_exp
            },
            noise_range=None,
            noise_need_action = True
        )
        self._collect_model = model_wrap(
            self._collect_model, wrapper_name='hidden_state_ac', state_num=1, save_prev_state=True
        )
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float, **kwargs) -> dict:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        # print("data before collected:", data)
        data = default_collate([data[i]['agent_state'] for i in data_id])
        # print("data after collected:", data)
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': {'agent_state': data}}
        self._collect_model.eval()
        with torch.no_grad():
            # in collect phase, inference=True means that each time we only pass one timestep data,
            # so the we can get the hidden state of rnn: <prev_state> at each timestep.
            model_output = self._collect_model.forward(data, mode="compute_actor", 
                                                 inference=True, **kwargs)
        if self._cuda:
            model_output = to_device(model_output, 'cpu')
        # eliminate the batch dim
        if model_output['prev_state']:
            for i in range(len(model_output['prev_state'])):
                model_output['prev_state'][i] = model_output['prev_state'][i][0]
        action_output = default_decollate(model_output['action'])
        trigger_output = default_decollate(model_output['trigger'])
        output = {i: {'action': a, 'trigger': t, 'prev_state': model_output['prev_state']} for i, a, t in zip(data_id, action_output, trigger_output)}
        return output

    def _reset_collect(self, *args, **kwargs) -> None:
        self._collect_model.reset(*args, **kwargs)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action', 'prev_state']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'action': model_output['action'],
            'trigger': model_output['trigger'],
            'prev_state': model_output['prev_state'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        # print("transition: ", transition)
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data

        Arguments:
            - data (:obj:`list`): The trajectory's cache

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        new_data = get_train_sample_spine(data, self._unroll_len_add_burnin_step, overlap = self._unroll_overlap)

        for sample in new_data:
            if sample['prev_state']:
                sample['prev_state'] = sample['prev_state'][0]
        return new_data
        

    #TODO: Check
    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='hidden_state_ac', state_num = 1)
        self._eval_model.reset()
        self._eval_model.set_slope(self._cfg.learn.init_slope)

    #TODO: Check
    def _forward_eval(self, data: dict, **kwargs) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        # print("data before collate in eval:", data)
        data = default_collate([data[i]['agent_state'] for i in data_id])
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': {'agent_state': data}}
        # print("data after collate in eval:", data)
        self._eval_model.eval()
        with torch.no_grad():
            model_output = self._eval_model.forward(data, mode="compute_actor", inference=True, **kwargs)
        if self._cuda:
            model_output = to_device(model_output, 'cpu')
        # eliminate the batch dim
        # print("the model output in eval:", model_output)
        # if 'prev_state' in model_output.keys() and model_output['prev_state']:
        #     for i in range(len(model_output['prev_state'])):
        #         model_output['prev_state'][i] = model_output['prev_state'][i][0]
        # else:
        #     model_output['prev_state'] = 
        action_output = default_decollate(model_output['action'])
        trigger_output = default_decollate(model_output['trigger'])
        output = {i: {'action': a, 'trigger': t} for i, a, t in zip(data_id, action_output, trigger_output)}
        return output

    def _reset_eval(self, *args, **kwargs) -> None:
        self._eval_model.reset(**kwargs)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'spine_model', ['train.policy.spine']

    def _monitor_vars_learn(self) -> List[str]:
        ret = [
            'cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'q_value_t0', 'q_value_t0_twin', 'td_error', 'trigger',
                        'actor_f_1', 
            'actor_f_2',
            'actor_i_1',
            'actor_i_2',
            'actor_o_1',
            'actor_o_2',
            'critic_f_1',
            'critic_i_1',
            'critic_o_1',
        ]
        if self._twin_critic:
            ret += ['critic_twin_loss']
        return ret


@POLICY_REGISTRY.register('spine_policy_command')
class SpineCommandModePolicy(SpinePolicy, EpsCommandModePolicy):
    pass