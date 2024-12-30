import math
import numpy as np
import random
from . import context
import random
from datetime import datetime
from helpers.logger import logger

STATE_DIM = 10
ACTION_DIM = 4

def is_global_state(state):
    if type(list(state.values())[0]) == dict:
        return True
    else:
        return False

def transform_state(state, env = None, flow_id = None, current_policy = None, num_flow = None):
    def state_dict_2_array(state_dict, current_policy = None):
        state = []    

        max_tput = state_dict["max_tput"]
        
        if state_dict["avg_thr"] == 0:
            state.append(-0.1)
        else:
            state.append(state_dict["avg_thr"] / max_tput if max_tput > 0 else 0)
            
        if state_dict["avg_urtt"] == 0:
            state.append(2)
        elif state_dict["min_rtt"]==0:
            state.append(-0.1)
        else:
            state.append(state_dict["min_rtt"]/state_dict["avg_urtt"])
            
        if state_dict["srtt_us"] == 0:
            state.append(2)
        elif state_dict["min_rtt"]==0:
            state.append(0)
        else:
            state.append(state_dict["min_rtt"]/(state_dict["srtt_us"] / 8))
               
        if state_dict["min_rtt"]==0 or max_tput==0:
            state.append(0)
        else:
            state.append(state_dict["cwnd"] * 1460 * 8 / (state_dict["min_rtt"] / 1e6 * max_tput) / 10)
        #print(state_dict["max_tput"]) 
        state.append(max_tput / 5e8)
        state.append(state_dict["min_rtt"] / 5e5)
        state.append(state_dict["loss_bytes"] / max_tput if max_tput>0 else 0)
        state.append(state_dict["packets_out"] / state_dict["cwnd"])
        state.append(state_dict["pacing_rate"] / max_tput if max_tput > 0 else 0)
        state.append(state_dict["retrans_out"] / state_dict["cwnd"])
        if current_policy is not None:
            state = state + current_policy
        return state

    
    if is_global_state(state):
        assert(env is not None)
        local_state = state_dict_2_array(state[flow_id], current_policy = current_policy)
        state_len = len(local_state)
        all_states = [state_dict_2_array(state[s], current_policy = current_policy) if s in state.keys() else [0] * state_len for s in range(num_flow) if s is not flow_id ]
        global_state = local_state + [len(state.keys())/10,  #sum(all_states, local_state) + [len(state.keys())/10, 
                        env.world.one_way_delay/100.0, 
                        env.world.bdp,
                        env.world.bandwidth/1000]
        return local_state, global_state
    else:
        return state_dict_2_array(state, current_policy= current_policy), []

def max_action(env):
    max_window = env.world.bandwidth * 1e6 * (2 * env.world.one_way_delay/1e3) / (1460 * 8)
    return int(max_window * 1.5)


def map_scubic_action(action, state):
    action = {
        'cubic_bic_scale': int((action[0] + 1) * 512.0),
        'cubic_beta':  512 + int((action[1] + 1) * 256.0) # change back to 1 remenber when two actions are used!
    }
    if action['cubic_bic_scale'] > 1000: 
        action['cubic_bic_scale'] = 1000 
    if action['cubic_bic_scale'] < 5:
        action['cubic_bic_scale'] = 5
    if action['cubic_beta'] >= 1024:   
        action['cubic_beta'] = 1023
    return action


def map_vanilla_action(action, state = None):
    action_state = {}
    action_state["vanilla_alpha"] =  int((action[0] + 1) * 128.0) # maximum 20%
    action_state["vanilla_beta"] =  int((action[1] + 1) * 128.0) # maximum 20%
    action_state["vanilla_gamma"] =  int((action[2] + 1) * 512.0) # maximum 100% (*2 in the kernel->2BDP buffer)
    action_state["vanilla_delta"] =  800 + int((action[3] + 1) * 112.0) # about 80%~100%
    for key in action_state:
        if action_state[key] <= 0:
            action_state[key] = 1
        elif action_state[key] >= 1024:
            action_state[key] = 1023
    return action_state
    
def default_vanilla_action():
    return {
        'vanilla_alpha': 100,
        'vanilla_beta': 100,
        'vanilla_gamma': 100,
        'vanilla_delta': 717
    }

# Spine's reward
TRIGGER_FACTOR = 0.001
def reward_callback(states_history, flow_id, world, env, trigger = 0):
    states = states_history[-1]  # the latest state
    throughputs = []
    losses = []
    rtts = []
    for fid in states.keys():
        if fid == flow_id:
            my_max_throughput = states[fid]["max_tput"]
            my_throughput = states[fid]["avg_thr"]
            my_lat =  states[fid]["avg_urtt"] / 1000
            my_loss = states[fid]["loss_bytes"]
        throughputs.append(states[fid]["avg_thr"])
        losses.append(states[fid]["loss_bytes"])
        rtt = states[fid]["avg_urtt"] / 1000
        if rtt > 0:
            rtts.append(rtt)
    loss = np.mean(losses)
    latency = np.mean(rtts)
    overall_throughput = sum(throughputs)
    if overall_throughput == 0:
        return 0, 0, 0, 0, 0
    
    # processing fairness
    if len(throughputs) == 1:
        fairness = 0
    else:
        fairness = - (np.std(throughputs)/np.sum(throughputs))
        if math.isnan(fairness):
            print("fairness is nan, reset to 0.0")
            fairness = 0.0
    fairness = fairness * overall_throughput / (env.world.bandwidth * 1e6 / 8)
    
    # processing phase
    if latency < 1.15 * (2 * (env.world.one_way_delay)):
        latency = 1
    else:
        latency = latency / (1.15 * 2 * env.world.one_way_delay)
    power = 0.1 * (0.5 * fairness + ((overall_throughput - 5 * 8 * loss) /  (env.world.bandwidth * 1e6 / 8)) / latency - TRIGGER_FACTOR * trigger)
    return power, my_max_throughput, loss, trigger, latency
