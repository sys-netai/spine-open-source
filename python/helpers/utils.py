import os
from os import path
from shutil import Error
import sys
import socket
import signal
import errno
import json
from traceback import print_exception
import traceback
from numpy.core.fromnumeric import trace
import yaml
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy

np.set_printoptions(threshold=np.inf)
from operator import add
from . import context
from helpers.subprocess_wrappers import check_call, check_output, call
from helpers.logger import logger

def loop_world_generator(world_config):
    # generate the world in a loop
    trace_list = world_config["trace_list"]
    delay_list = world_config["delay_list"]
    # first bw, then delay, then buffer, then loss.
    for base_delay in delay_list:
        for bw in trace_list:
            bw_str = "%dmbps.trace"%bw
            if "bdp_list" in world_config.keys():
                bdp_list = world_config["bdp_list"]
                uplink_queue_args_list = bw * base_delay * 2 * 1e3 / 8 * np.array(bdp_list)
                uplink_queue_args_list = uplink_queue_args_list.astype(int)
            elif "uplink_queue_args_list" in world_config.keys():
                uplink_queue_args_list = world_config['uplink_queue_args_list']
            for uplink_queue_args in uplink_queue_args_list:
                bdp = uplink_queue_args / (bw * base_delay * 2 * 1e3 / 8)
                if "random_loss_list" in world_config.keys():
                    random_loss_list = world_config['random_loss_list']
                else:
                    random_loss_list = [0]
                for random_loss in random_loss_list:
                    # still random cnoose the variance and extra_delay
                    if "extra_delay" in world_config.keys():
                        extra_delays = dict()
                        for i in range(world_config["flows"]):
                            extra_delays[str(i)] = world_config["extra_delay"][i]
                    else:
                        extra_delays = dict()
                        for i in range(world_config["flows"]):
                            extra_delays[str(i)] = random.choice(world_config["extra_delay_list"])
                    
                    
                    new_world = copy.deepcopy(world_config)
                    new_world.pop("trace_list", None)
                    new_world.pop("extra_delay_list", None)
                    new_world.pop("delay_list", None)
                    new_world.pop("bdp_list", None)
                    new_world.pop("random_loss_list", None)
                    new_world.pop("uplink_queue_args_list", None)
                    if "variance_freq_list" in world_config.keys():
                        variance_freq = random.choice(world_config['variance_freq_list'])
                        variance_range = random.choice(world_config['variance_range_list'])
                        new_world['variance_freq'] = variance_freq
                        new_world['variance_range'] = variance_range
                        
                    new_world["bdp"] = bdp
                    new_world["extra_delay"] = extra_delays
                    new_world["uplink_trace"] = bw_str
                    new_world["downlink_trace"] = bw_str
                    new_world["uplink_queue_args"] = "bytes=%d"%uplink_queue_args
                    new_world["one_way_delay"] = base_delay
                    new_world['random_loss'] = random_loss
                    yield new_world
    
    
    

def generate_world(world_config):
    trace_list = world_config["trace_list"]
    extra_delay_list = world_config["extra_delay_list"]
    delay_list = world_config["delay_list"]
    base_delay = random.choice(delay_list)
    bw = random.choice(trace_list)
    bw_str = "%dmbps.trace"%bw
    if "bdp_list" in world_config.keys():
        bdp_list = world_config["bdp_list"]
        bdp = random.choice(bdp_list)
        uplink_queue_args = int(bw * base_delay * 2 * 1e3 / 8 * bdp)
    elif "uplink_queue_args_list" in world_config.keys():
        uplink_queue_args = random.choice( world_config['uplink_queue_args_list'])
        bdp = uplink_queue_args / (bw * base_delay * 2 * 1e3 / 8)
    extra_delays = dict()
    for i in range(world_config["flows"]):
        extra_delays[i] = random.choice(extra_delay_list)
    
    new_world = copy.deepcopy(world_config)
    if "all" in new_world["flow_cc"] and new_world["flow_cc"]["all"] == "spine":
        num_our_flows = new_world["flows"]
    else:
        num_our_flows = 0
        for flow_id in new_world["flow_cc"].keys():
            if new_world["flow_cc"][flow_id] == "spine":
                num_our_flows += 1
    new_world["num_our_flows"] = num_our_flows
    if "variance_freq_list" in world_config.keys():
        variance_freq = random.choice(world_config['variance_freq_list'])
        variance_range = random.choice(world_config['variance_range_list'])
        new_world['variance_freq'] = variance_freq
        new_world['variance_range'] = variance_range
    new_world["bdp"] = bdp
    new_world["uplink_trace"] = bw_str
    new_world["downlink_trace"] = bw_str
    new_world["uplink_queue_args"] = "bytes=%d"%uplink_queue_args
    new_world["extra_delay"] = extra_delays
    new_world["one_way_delay"] = base_delay
    if "random_loss_list" in world_config.keys():
        random_loss = random.choice(world_config['random_loss_list'])
        new_world['random_loss'] = random_loss
    
    print(new_world)
    return new_world

def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            sys.stderr.write(traceback.format_exec())
            print_exception(e)

    return inner_function


def fill_array_with_dict(target_dict, idx_required=False):
    if len(target_dict.items()) == 0:
        if idx_required:
            return [[0], [0]]
        else:
            return [0]
    sorted_items = sorted(target_dict.items())
    idx, values = list(zip(*sorted_items))
    if idx_required:
        return np.array(idx), np.array(values)
    else:
        return np.array(values)


def take_a_photo_cwnd(
    thr_histories,
    act_histories,
    given_act_histories,
    lat_histories,
    reward_histories,
    metric1_histories,
    metric2_histories,
    metric3_histories,
    metric4_histories,
    env,
    filename,
):
    if len(reward_histories) == 0:
        print("empty photo!")
        return
    print("take a photo!")
    reward_xs, reward_ys = fill_array_with_dict(reward_histories, True)
    metric1_histories = fill_array_with_dict(metric1_histories)
    metric2_histories = fill_array_with_dict(metric2_histories)
    metric3_histories = fill_array_with_dict(metric3_histories)
    metric4_histories = fill_array_with_dict(metric4_histories)
    flow_xs = [None] * env.world.num_our_flows
    thr_ys = [None] * env.world.num_our_flows
    act_ys = [None] * env.world.num_our_flows
    lat_ys = [None] * env.world.num_our_flows

    given_act_ys = [None] * env.world.num_our_flows
    given_act_xs = [None] * env.world.num_our_flows

    for i in range(env.world.num_our_flows):
        # print("histories:", thr_histories[i],act_histories[i],lat_histories[i])
        flow_xs[i], thr_ys[i] = fill_array_with_dict(thr_histories[i], True)
        given_act_xs[i], given_act_ys[i] = fill_array_with_dict(
            given_act_histories[i], True
        )
        act_ys[i] = fill_array_with_dict(act_histories[i])
        lat_ys[i] = fill_array_with_dict(lat_histories[i])
        # print(i)
        # print("thr_ys:",thr_ys[i])
        # print("act_ys:",act_ys[i])
        # print("lat_ys:",lat_ys[i])
    sum_x = np.unique(np.concatenate([flow_xs[i] for i in range(env.world.num_our_flows)]))
    ys_sum = np.zeros_like(sum_x)

    # rendering
    fig, axs = plt.subplots(4, 2, figsize=(16, 20))
    axs[0, 0].set_title("throughput")
    axs[1, 0].set_title("cwnd")
    axs[2, 0].set_title("latency")
    axs[3, 0].set_title("reward")
    axs[0, 1].set_title("latency-reward")
    axs[1, 1].set_title("thr-reward")
    axs[2, 1].set_title("fairness-reward")
    axs[3, 1].set_title("stability-reward")
    for i in range(env.world.num_our_flows):
        ys_sum += np.interp(sum_x, flow_xs[i], thr_ys[i], left=0, right=0)
        axs[0, 0].plot(flow_xs[i], thr_ys[i])
        axs[1, 0].plot(flow_xs[i], act_ys[i])
        axs[1, 0].plot(given_act_xs[i], given_act_ys[i], linewidth=0.5)
        axs[2, 0].plot(flow_xs[i], lat_ys[i])

    axs[3, 0].plot(reward_xs, reward_ys)
    axs[0, 1].plot(reward_xs, metric1_histories)
    axs[1, 1].plot(reward_xs, metric2_histories)
    axs[2, 1].plot(reward_xs, metric3_histories)
    axs[3, 1].plot(reward_xs, metric4_histories)
    # plot capacity
    capacity_x = np.arange(0,math.ceil(sum_x[-1]), 0.1)
    capacity_y = np.array(env.world.bandwidth_list) * 1e6 / 8
    (line,) = axs[0, 0].plot(capacity_x, capacity_y[:len(capacity_x)], color="black", linewidth=1)
    line.set_label("capacity")
    
    # plot sum bandwidth
    (line,) = axs[0, 0].plot(sum_x, ys_sum, color="red", linewidth=2)
    line.set_label("sum")
    axs[0, 0].legend()
    plt.savefig(filename + "%dbw-%dd-%dbdp-%.3floss.png" % (env.world.bandwidth, env.world.one_way_delay, env.world.bdp, env.world.random_loss), dpi=300)


def take_a_photo_policy(
    thr_histories,
    act_histories,
    given_act_histories,
    lat_histories,
    reward_histories,
    metric1_histories,
    metric2_histories,
    metric3_histories,
    metric4_histories,
    env,
    filename,
):
    if len(reward_histories) == 0:
        print("empty photo!")
        return
    print("take a photo!")
    flow_xs = [None] * env.world.num_our_flows
    thr_ys = [None] * env.world.num_our_flows
    act_ys = [None] * env.world.num_our_flows
    lat_ys = [None] * env.world.num_our_flows
    
    reward_ys = [None] * env.world.num_our_flows
    metric1_ys = [None] * env.world.num_our_flows
    metric2_ys = [None] * env.world.num_our_flows
    metric3_ys = [None] * env.world.num_our_flows
    metric4_ys = [None] * env.world.num_our_flows
    

    sampled_action = list(given_act_histories[0].values())[0]
    action_dim = len(sampled_action)
    # print("!!!", given_act_histories[0])
    # print("!!!", given_act_histories[1])
    assert action_dim < 5
    given_act_ys = [[None for i in range(env.world.num_our_flows)] for i in range(action_dim)]
    given_act_xs = [[None for i in range(env.world.num_our_flows)] for i in range(action_dim)]

    for i in range(env.world.num_our_flows):
        # print("histories:", thr_histories[i],act_histories[i],lat_histories[i])
        flow_xs[i], thr_ys[i] = fill_array_with_dict(thr_histories[i], True)
        act_ys[i] = fill_array_with_dict(act_histories[i])
        lat_ys[i] = fill_array_with_dict(lat_histories[i])
        reward_ys[i] = fill_array_with_dict(reward_histories[i])
        metric1_ys[i] = fill_array_with_dict(metric1_histories[i])
        metric2_ys[i] = fill_array_with_dict(metric2_histories[i])
        metric3_ys[i] = fill_array_with_dict(metric3_histories[i])
        metric4_ys[i] = fill_array_with_dict(metric4_histories[i])
        
        for j in range(action_dim):
            policy_action = {key: given_act_histories[i][key][j] for key in given_act_histories[i]}
            # print("policy_action of j:", j,  policy_action)
            given_act_xs[j][i], given_act_ys[j][i] = fill_array_with_dict(
                policy_action, True
            )
        # print(i)
        # print("thr_ys:",thr_ys[i])
        # print("act_ys:",act_ys[i])
        # print("lat_ys:",lat_ys[i])
    sum_x = np.unique(np.concatenate([flow_xs[i] for i in range(env.world.num_our_flows)]))
    ys_sum = np.zeros_like(sum_x)

    # rendering
    fig, axs = plt.subplots(4, 3, figsize=(24, 20))
    axs[0, 0].set_title("throughput")
    axs[1, 0].set_title("cwnd")
    axs[2, 0].set_title("latency")
    axs[3, 0].set_title("reward")
    
    axs[0, 1].set_title("metric1")
    axs[1, 1].set_title("metric2")
    axs[2, 1].set_title("metric3")
    axs[3, 1].set_title("metric4")
    
    axs[0, 2].set_title("policy-action1")
    axs[1, 2].set_title("policy-action2")
    axs[2, 2].set_title("policy-action3")
    axs[3, 2].set_title("policy-action4")
    for i in range(env.world.num_our_flows):
        ys_sum += np.interp(sum_x, flow_xs[i], thr_ys[i], left=0, right=0)
        axs[0, 0].plot(flow_xs[i], thr_ys[i])
        axs[1, 0].plot(flow_xs[i], act_ys[i])
        axs[2, 0].plot(flow_xs[i], lat_ys[i])
        for j in range(len(given_act_xs)):
            axs[j, 2].plot(given_act_xs[j][i], given_act_ys[j][i])
        axs[3, 0].plot(flow_xs[i], reward_ys[i])
        axs[0, 1].plot(flow_xs[i], metric1_ys[i])
        axs[1, 1].plot(flow_xs[i], metric2_ys[i])
        axs[2, 1].plot(flow_xs[i], metric3_ys[i])
        axs[3, 1].plot(flow_xs[i], metric4_ys[i])
    # plot capacity
    capacity_x = np.arange(0,math.ceil(sum_x[-1]), 0.1)
    capacity_y = np.array(env.world.bandwidth_list) * 1e6 / 8
    (line,) = axs[0, 0].plot(capacity_x, capacity_y[:len(capacity_x)], color="black", linewidth=1)
    line.set_label("capacity")
    # plot sum bandwidth
    (line,) = axs[0, 0].plot(sum_x, ys_sum, color="red", linewidth=2)
    line.set_label("sum")
    axs[0, 0].legend()
    plt.savefig(filename + "~{0}bw-{1}d-{2:.1f}bdp-{3:.3f}loss.png".format(env.world.bandwidth, env.world.one_way_delay, env.world.bdp, env.world.random_loss), dpi=300)
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')


def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)


class Params:
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def sample_action(alg, state, tun_id=None):
    if tun_id == 1:
        return 200
    else:
        return 400


def check_kernel_module(cc_name):
    if call("sudo modprobe tcp_{}".format(cc_name), shell=True) != 0:
        sys.exit("kernel module tcp_{} is not available".format(cc_name))


def prepare_kernel_tcp_cc(cc_name):
    ccs = check_output("sysctl net.ipv4.tcp_allowed_congestion_control", shell=True)
    cc_list = ccs.split("=")[-1].split()

    if cc_name in cc_list:
        return

    check_kernel_module(cc_name)

    cc_list.append(cc_name)
    check_call(
        'sudo sysctl -w net.ipv4.tcp_allowed_congestion_control="{}"'.format(
            " ".join(cc_list)
        ),
        shell=True,
    )
    logger.info("Load TCP:{} to kernel".format(cc_name))


def kill_iperf():
    os.system("pkill -9 iperf")


def get_open_port():
    sock = socket.socket(socket.AF_INET)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return str(port)


def make_sure_dir_exists(d):
    try:
        os.makedirs(d)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


tmp_dir = path.join(context.base_dir, "tmp")
make_sure_dir_exists(tmp_dir)


def parse_config():
    with open(path.join(context.src_dir, "config.yml")) as config:
        return yaml.load(config)


def update_submodules():
    cmd = "git submodule update --init --recursive"
    check_call(cmd, shell=True)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError()


def utc_time():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def kill_proc_group(proc, signum=signal.SIGTERM):
    if not proc:
        return

    try:
        logger.info(
            "kill_proc_group: killed process group with pgid %s\n"
            % os.getpgid(proc.pid)
        )
        os.killpg(os.getpgid(proc.pid), signum)
    except OSError as exception:
        logger.error("kill_proc_group: %s\n" % exception)


def query_clock_offset(ntp_addr, ssh_cmd):
    local_clock_offset = None
    remote_clock_offset = None

    ntp_cmds = {}
    ntpdate_cmd = ["ntpdate", "-t", "5", "-quv", ntp_addr]

    ntp_cmds["local"] = ntpdate_cmd
    ntp_cmds["remote"] = ssh_cmd + ntpdate_cmd

    for side in ["local", "remote"]:
        cmd = ntp_cmds[side]

        fail = True
        for _ in range(3):
            try:
                offset = check_output(cmd)
                sys.stderr.write(offset)

                offset = offset.rsplit(" ", 2)[-2]
                offset = str(float(offset) * 1000)
            except subprocess.CalledProcessError:
                sys.stderr.write("Failed to get clock offset\n")
            except ValueError:
                sys.stderr.write("Cannot convert clock offset to float\n")
            else:
                if side == "local":
                    local_clock_offset = offset
                else:
                    remote_clock_offset = offset

                fail = False
                break

        if fail:
            sys.stderr.write("Failed after 3 queries to NTP server\n")

    return local_clock_offset, remote_clock_offset
