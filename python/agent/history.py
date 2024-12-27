import os
import sys
import json
import math
import threading
import numpy as np
from . import context

from helpers.logger import logger

STATE_DIM = 10
ACTION_DIM = 1
GLOBAL_DIM = 12


def compute_reward(states: dict, world_size, config):
    return 0


def parse_states(input: dict):
    state = np.zeros(STATE_DIM)
    # min_rtt in us
    state[0] = int(input["min_rtt"])
    # avg_rtt in us since last info request
    state[1] = int(input["avg_urtt"])
    # throughput bytes per second
    state[2] = int(input["avg_thr"])
    # max throughput has observed
    state[3] = int(input["max_tput"])
    # number of rate samples used for calculating throughput
    state[4] = int(input["thr_cnt"])
    # pacing rate
    state[5] = int(input["pacing_rate"])
    # smoothed RTTs in us << 3
    state[6] = int(input["srtt_us"] / 8)
    # lost bytes
    state[7] = int(input["loss_bytes"])
    # current cwnd
    state[8] = int(input["cwnd"])

    return state


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_json"):
            return obj.to_json()
        else:
            return json.JSONEncoder.default(self, obj)


class Transition(object):
    def __init__(self, flow_id, state, action, reward=None, done=None):
        self.raw_states = {}
        self.states = {}
        self.action = None
        self.reward = reward
        self.next_state = None

        self.world_size = 0
        self.observed_size = 0

        self.add_world_state(flow_id, state)
        self.set_action(action)

    def add_world_state(self, flow_id, state):
        if state is not None:
            if flow_id in self.states:
                raise Exception("Flow {} has already been added".format(flow_id))
            self.raw_states[flow_id] = state
            # we need to process the input, change dict the np.array()
            # self.states[flow_id] = parse_states(state)
            self.observed_size += 1

    def set_action(self, action):
        if action != None:
            self.action = action

    def set_reward(self, reward):
        if reward != None:
            self.reward = reward

    def set_world_size(self, world_size):
        self.world_size = world_size

    def to_json(self):
        return self.__dict__

    def ready(self):
        if self.observed_size == self.world_size:
            return True
        else:
            return False

    def get_states(self):
        states = {}
        # we want to keep the order
        # the state of flow-i should be in states[i-1]
        for flow_id, state in sorted(self.states.items()):
            states[flow_id] = state
        return states

    def get_world_size(self):
        return self.world_size


class FlowState(object):
    """Class for saving world state

    This class is used by any single flow to observe the world.
    """

    def __init__(self, flow_id):
        self.flow_id = flow_id
        self.step = 0
        # key is step, val is Transition
        self.transitions = {}

    def add_world_state(self, step, flow_id, state):
        self.transitions[step].add_world_state(flow_id, state)

    def make_step_transition(self, flow_id, state, action):
        self.transitions[self.step] = Transition(flow_id, state, action)
        current_step = self.step
        self.step += 1
        return current_step

    def set_step_action(self, step_id, action):
        self.transitions[step_id].set_action(action)

    def get(self, step):
        return self.transitions[step]

    def set_world_size(self, step, world_size):
        self.transitions[step].set_world_size(world_size)

    def to_json(self):
        return self.__dict__


class History(object):
    def __init__(self, episode, config: dict):
        self.episode = episode
        self.config = config
        self.history = {}

    def add_flow_step(self, flow_id, state, action=None, reward=None, done=None):
        if flow_id not in self.history:
            self.history[flow_id] = FlowState(flow_id)

        step = self.history[flow_id].make_step_transition(flow_id, state, action)
        return step

    def add_world_state(self, observer_id, step_id, flow_id, state):
        self.history[observer_id].add_world_state(step_id, flow_id, state)

    def add_flow_step_action(self, observer_id, step_id, action):
        self.history[observer_id].set_step_action(step_id, action)

    def make_transitions(self):
        # formulate transition as <state, action, reward, next_state>
        pass

    def get(self, flow_id, step_id):
        transition: Transition = self.history.get(flow_id).get(step_id)
        if not transition.ready():
            # maybe one flow has terminated
            logger.warning(
                "Flow {}: step {} was not ready, current collected num: {}, world_num: {}".format(
                    flow_id, step_id, transition.observed_size, transition.world_size
                )
            )

        return transition.raw_states, transition.action, transition.reward

    def set_flow_step_world_size(self, observer_id, step, world_size):
        self.history[observer_id].set_world_size(step, world_size)

    def flow_step_state_ready(self, flow_id, step_id, current_world_size=None):
        trans: Transition = self.history.get(flow_id).get(step_id)
        if trans.ready():
            # calculate reward
            trans.set_reward(
                compute_reward(trans.get_states(), trans.world_size, self.config)
            )
            return True
        elif current_world_size != None and trans.observed_size >= current_world_size:
            return True
        else:
            return False

    def to_json(self):
        return self.__dict__
