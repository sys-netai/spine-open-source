#!/usr/bin/env python3

import argparse
import sys
import os
from os import path
import numpy as np
import json
import signal
import math
import torch
import matplotlib
from easydict import EasyDict
import random

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import time

from ding.config import read_config, compile_config
from ding.policy import Policy, create_policy

import context
from context import src_dir
from agent.definitions import transform_state, map_vanilla_action, map_vanilla_action_with_noise
from helpers.utils import Params
from helpers.logger import logger
from helpers.ipc_socket import IPCSocket
from helpers.poller import PollEvents, ReturnStatus, Action, Poller
from helpers.message import MessageType
torch.set_num_threads(1)


from helpers.total_config import exp_config
#from model.backup.oct_7_max_known.total_config import exp_config

model_path = path.abspath(
    path.join(
        src_dir, "train/config/serial/vanilla_train_seed0/ckpt/ckpt_best_21000.pth.tar"
        #"model/backup/oct_7_max_known", "ckpt", "ckpt_best_28000.pth.tar"
    )
)
spine_ipc_path = "/tmp/spine_ipc"

def main():
    sys.stdout = sys.__stderr__
    sys.stderr.write("[spine pyhelper] starting spine inference process\n")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default = model_path, help="path of saved models"
    ) # in fact is not used.
    parser.add_argument(
        "--ipc-path", type=str, help="IPC path of communication", required=True
    )
    # this is to identify each flow
    parser.add_argument(
        "--dst-port", type=int, required=True, help="destination port of current sender"
    )

    parser.add_argument(
        "--spine-ipc-path",
        type=str,
        help="IPC path of communication between SPINE kernel helper",
        default=spine_ipc_path,
    )

    args = parser.parse_args()

    # create_config.policy.type += "_command"
    # cfg = compile_config(
    #     main_config,
    #     seed=0,
    #     env=None,
    #     auto=True,
    #     create_cfg=create_config,
    #     save_cfg=True,
    #     save_path="eval_config.py",
    # )
    cfg = EasyDict(exp_config)
    # # Prepare policy
    policy = create_policy(cfg.policy, enable_field=["eval"])
    sys.stderr.write(f"PyHelper: Loading model from: {model_path}\n")
    #state_dict = torch.load(args.model_path, map_location="cpu") # not used!
    state_dict = torch.load(model_path, map_location="cpu")
    policy.eval_mode.load_state_dict(state_dict)
    eval_policy = policy.eval_mode

    # connect to C++ flow module
    client_ipc = IPCSocket()
    client_ipc.connect(args.ipc_path)

    # connect to Spine kernel helper (spine_eval.py)
    spine_ipc = IPCSocket()
    spine_ipc.connect(args.spine_ipc_path)

    def signal_handler(signum, frame):
        if signum == signal.SIGINT or signum == signal.SIGTERM:
            msg = {}
            msg["type"] = MessageType.END.value
            msg["dst_port"] = args.dst_port
            spine_ipc.write(json.dumps(msg))

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # send init message
    msg = {}
    msg["type"] = MessageType.START.value
    msg["dst_port"] = args.dst_port
    spine_ipc.write(json.dumps(msg))
    last_action = [0,0,0,0]
    last_trigger = 0
    # def process_state():
    while True:
        info = client_ipc.read()
        info = json.loads(info)
        state = info["state"]
        obs, _ = transform_state(state, current_policy = last_action) #, last_trigger = last_trigger)
        obs = {0: {"agent_state": torch.Tensor(obs)}}
        policy_output = eval_policy.forward(obs)[0]
        output = {
            key: o.numpy() if type(o) == torch.Tensor else o
            for key, o in policy_output.items()
        }
        last_trigger = output["trigger"][0]
        if last_trigger != 0:
            a = output["action"].flatten().tolist()
            last_action = a
            a = map_vanilla_action(a)
            reply = {}
            reply["type"] = MessageType.ALIVE.value
            reply["dst_port"] = args.dst_port
            reply["action"] = a
        
        # action_state = {}
        # action_state["vanilla_alpha"] =  100
        # action_state["vanilla_beta"] =  500
        # action_state["vanilla_gamma"] =  100
        # action_state["vanilla_delta"] =  717
        # logger.info("RL: action is {}".format(act))
            spine_ipc.write(json.dumps(reply))

        # return ReturnStatus.Continue

    #
    # poller = Poller()
    # poller.add_action(Action(client_ipc, PollEvents.READ_ERR_FLAGS, process_state))
    #
    # while True:
    #     poller.poll_once()
        
if __name__ == "__main__":
    main()
