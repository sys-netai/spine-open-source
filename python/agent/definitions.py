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
    # return a array with transformed states
    # print(f"flow: {flow_id}, state:{state}")
    def state_dict_2_array(state_dict, current_policy = None):
        #print(state_dict["max_tput"])
        state = []
        # 无量纲
        # state.append(state_dict["avg_thr"] / state_dict["max_tput"] if state_dict["max_tput"] > 0 else 0)
        # if state_dict["avg_urtt"] == 0:
        #     state.append(3)
        # elif state_dict["min_rtt"]==0:
        #     state.append(0)
        # else:
        #     state.append(state_dict["avg_urtt"]  / state_dict["min_rtt"] - 1)
        # # state.append(state_dict["cwnd"] / 1000)
        # if state_dict["min_rtt"]==0 or state_dict["max_tput"]==0:
        #     state.append(0)
        # else:
        #     state.append(state_dict["cwnd"] / 1000 /  (state_dict["max_tput"] / 1e7  *  state_dict["min_rtt"] / 1e5))
        # state.append(state_dict["loss_bytes"] / 1e6)
        # state.append(state_dict["pacing_rate"] / state_dict["max_tput"] if state_dict["max_tput"] > 0 else 1)
        # state.append(state_dict["packets_out"] / state_dict["cwnd"] / 10)
        # 混合型
        
        
        max_tput = state_dict["max_tput"]
        # if env is not None:
        #     max_tput = env.world.bandwidth * 1e6 / 8
        # else:
        #     max_tput = 300 * 1e6 / 8
        
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
        # if state[2] > 3:
        #     state[2] = 3
        # if state[1] > 3:
        #     state[1] = 3
        # if state[3] > 3:
        #     state[3] = 3
        # if state[8] > 3:
        #     state[8] = 3
            
        
        # state.append(state_dict["avg_thr"] / 1e7)
        # state.append(state_dict["avg_urtt"] / 1e5 if state_dict["avg_urtt"]>1000 else 10)
        # state.append(state_dict["cwnd"] / 1000)
        # state.append(state_dict["loss_bytes"] / 1e6)
        # state.append(state_dict["max_tput"] / 1e7)
        # state.append(state_dict["min_rtt"] / 1e5)
        # state.append(state_dict["pacing_rate"] / 1e7)
        # state.append(state_dict["packets_out"] / state_dict["cwnd"])
        # state.append(state_dict["srtt_us"] / 1e5 / 8 if state_dict["srtt_us"]>1000 else 10)
        # logger.error(state)
        # logger.error(datetime.now())
        return state

    
    if is_global_state(state):
        assert(env is not None)
        local_state = state_dict_2_array(state[flow_id], current_policy = current_policy)
        state_len = len(local_state)
        all_states = [state_dict_2_array(state[s], current_policy = current_policy) if s in state.keys() else [0] * state_len for s in range(num_flow) if s is not flow_id ]
        # thr = np.sum([state[s]["avg_thr"]/1e8 for s in state.keys()])
        # min_thr = np.min([state[s]["avg_thr"]/1e8 for s in state.keys()])
        # max_thr = np.max([state[s]["avg_thr"]/1e8 for s in state.keys()])
        # lat = np.mean([state[s]["avg_urtt"]/5e5 for s in state.keys()])
        # min_window = np.min([state[s]["cwnd"]/1000 for s in state.keys()])
        # max_window = np.max([state[s]["cwnd"]/1000 for s in state.keys()])
        # mean_window = np.mean([state[s]["cwnd"]/1000 for s in state.keys()])
        # loss = np.mean([state[s]["loss_bytes"]/1e6 for s in state.keys()])
        # num_flows = len(state.keys())/10
        # global_state = [thr, min_thr, max_thr, lat, min_window, max_window, mean_window, loss, num_flows, 
        #                 env.world.one_way_delay/100.0, 
        #                 env.world.bdp,
        #                 env.world.bandwidth/1000]
        global_state = sum(all_states, local_state) + [len(state.keys())/10, 
                        env.world.one_way_delay/100.0, 
                        env.world.bdp,
                        env.world.bandwidth/1000]      
        # print("local:", local_state)  
        # print("global:", global_state)
        return local_state, global_state
    else:
        #print("current policy:", current_policy)
        return state_dict_2_array(state, current_policy= current_policy), []

def max_action(env):
    max_window = env.world.bandwidth * 1e6 * (2 * env.world.one_way_delay/1e3) / (1460 * 8)
    return int(max_window * 1.5)


def map_cwnd_action(action, state):
    cwnd = state['cwnd']
    range_high = 0.025
    range_low = 0.025
    # if cwnd < int(1/range):
    if action >= 0:
        out = 1 + range_high * (action)
        out = math.ceil(out * cwnd)
    else:
        out = 1 / (1 - range_low * (action))
        out = math.floor(out * cwnd)
    # else:
    #     if action >= 0:
    #         out = (1 + range * (action)) * cwnd
    #     else:
    #         out = (1 / (1 - range * (action))) * cwnd
    #     frac = out - math.floor(out)
    #     if random.random() > frac:
    #         out = math.floor(out)
    #     else:
    #         out = math.ceil(out)
    return out

def map_vanilla_action(action, state = None):
    action_state = {}
    action_state["vanilla_alpha"] =  int((action[0] + 1) * 100.0) # maximum 100%
    action_state["vanilla_beta"] =  int((action[1] + 1) * 100.0) # maximum 100%
    action_state["vanilla_gamma"] =  int((action[2] + 1) * 512.0) # maximum 100% (*2 in the kernel)
    action_state["vanilla_delta"] =  800 + int((action[3] + 1) * 100.0)
    for key in action_state:
        if action_state[key] <= 0:
            action_state[key] = 1
        elif action_state[key] >= 1024:
            action_state[key] = 1023
    return action_state
    
def map_vanilla_action_with_noise(action, state = None):
    action_state = {}
    action_state["vanilla_alpha"] =  int((action[0] + 1) * 100.0) + random.randint(0,100) - 50 # maximum 100%
    action_state["vanilla_beta"] =  int((action[1] + 1) * 100.0) + random.randint(0,100) - 50 # maximum 100%
    action_state["vanilla_gamma"] =  int((action[2] + 1) * 512.0) + random.randint(0,100) - 50# maximum 100% (*2 in the kernel)
    action_state["vanilla_delta"] = 800 + int((action[3] + 1) * 100.0) + random.randint(0,100) - 50
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


# def reverse_action(action, cwnd):
#     if action == 0:
#         return 1
#     action /= cwnd
#     if action >= 1:
#         out = (action - 1) * 20
#     else:
#         out = (1 - 1 / action) * 20
#     if out < -1:
#         out = -1
#     if out > 1:
#         out = 1
#     return out

# astraea's reward
# TRIGGER_FACTOR = 0.001
# def reward_callback(states_history, flow_id, world, env, trigger = 0):
#     # print("state history for reward calculation:", states_history)
#     # ingress, latency, action
#     # reward: sub-metrics
#     action_variances = []
#     averaged_throughputs = []
#     my_variance = 0
#     for flow_id in range(world["flows"]):
#         action_sum = []
#         accumulated_throughput = []
#         for states in states_history:
#             if flow_id in states.keys():
#                 action_sum.append(states[flow_id]["cwnd"])
#                 # accumulated_throughput.append(states[flow_id]["avg_thr"])
#         if len(action_sum) > 1:
#             action_variances.append(np.std(action_sum))
#             # if flow_id == my_flow_id:
#             #     my_variance = np.std(action_sum)
#         # if len(accumulated_throughput) > 0:
#         #     averaged_throughputs.append(np.mean(accumulated_throughput))
#     if len(action_variances) > 0:
#         overall_window_variance = np.mean(action_variances)
#     else:
#         overall_window_variance = 0

#     states = states_history[-1]
#     active_flow_num = len(states.keys())
#     throughputs = []
#     pacing_rates = []
#     losses = []
#     rtts = []
#     my_throughput = 0
#     my_pacing_rate = 0
#     my_loss = 0
#     for flow_id in states.keys():
#         throughputs.append(states[flow_id]["avg_thr"])
#         pacing_rates.append(states[flow_id]["pacing_rate"])
#         losses.append(states[flow_id]["loss_ratio"]/states[flow_id]
#                       ["pacing_rate"] if not states[flow_id]["pacing_rate"] == 0 else 0)
#         # if flow_id == my_flow_id:
#         #     my_throughput = np.mean(states[flow_id]["avg_thr"])
#         #     my_pacing_rate = np.mean(states[flow_id]["pacing_rate"])
#         #     my_loss = (states[flow_id]["loss_ratio"]/states[flow_id]["avg_thr"] 
#         #                if not states[flow_id]["avg_thr"] == 0 else 0)
#         rtt = states[flow_id]["avg_urtt"] / 1000 - 2 * (
#             env.world.extra_per_flow_delay[flow_id] + env.world.one_way_delay
#         )
#         if rtt > 0:
#             rtts.append(rtt)
#     loss = np.mean(losses)
#     latency = np.mean(rtts)
#     latency -= 0.3 * env.world.one_way_delay
#     if latency < 0:
#         latency = 0
#     overall_throughput = sum(throughputs)
#     overall_pacing_rates = sum(pacing_rates)
#     if overall_throughput == 0:
#         return 0, 0, 0, 0, 0
#     if len(throughputs) == 1:
#         fairness = 1
#     else:
#         fairness = 1 - (np.std(throughputs) /
#                         np.sum(throughputs))
#         # fairness = float(
#         #     math.pow(np.sum(averaged_throughputs), 2) / (len(averaged_throughputs) * np.sum(np.power(averaged_throughputs, 2)))
#         # )
#         if math.isnan(fairness):
#             print("fairness is nan, reset to 0.0")
#             fairness = 0.0
#     if fairness < 0.5:
#         fairness = 0.5
#     # use pernaty regarding input to buffer, not output.
#     metric1 = overall_pacing_rates * latency / \
#         (env.world.bandwidth * 1e6 / 8) / env.world.one_way_delay / math.sqrt(active_flow_num)
#     metric2 = overall_throughput / (env.world.bandwidth * 1e6 / 8)
#     if metric2 > 1:
#         metric2 = 1
#     metric3 = (fairness - 1) * metric2
#     metric4 = overall_window_variance / 100
#     if metric4 > 1 or fairness < 0.96:
#         metric4 = 1
#     metric4 = (1 - metric4) * metric2
#     metric5 = loss
#     # print("latency", latency,
#     #       "overall_pacing_rates", overall_pacing_rates,
#     #       "overall_throughput", overall_throughput,
#     #       "fairness", fairness,
#     #       "overall_window_variance", overall_window_variance)
#     # print("metric1: ", metric1, "metric2: ", metric2, "metric3: ", metric3, "metric4:", metric4)
#     reward = - 0.02 * metric1 + 0.1 * metric2  + 0.01 * metric4 - 1 * metric5 - TRIGGER_FACTOR * trigger
#     if reward < -0.5:
#         reward = -0.5
#     return reward, - 0.02 * metric1, 0.1 * metric2, 0.01 * metric4, 1 * metric5

TRIGGER_FACTOR = 0.001
# Spine's reward
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
    
    # print("state for rewarding:", states, "reward:",power)
    return power, my_max_throughput, loss,  fairness, latency
