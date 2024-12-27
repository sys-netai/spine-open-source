import multiprocessing
import os
import sys
import json
import functools
import threading
import time
import copy
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import traceback
from . import context

from agent.history import History, ComplexEncoder
from env.message import MessageType
from helpers.ipc_socket import IPCSocket
from helpers.logger import logger
from helpers.poller import PollEvents, ReturnStatus, Action, Poller


class Master(object):
    def __init__(self, ipc_file, num_flow, dump_state=False):
        self.num_agents = num_flow
        # self.pool = ThreadPool(num_flow)
        self.pool = ThreadPool(num_flow)
        logger.info("Master: create {} threads in pool".format(num_flow))
        self.process_poll = Pool(num_flow)

        # communicate with env
        self.env_flow_id = 10000
        self.env_ipc = None
        self.poller = Poller()
        # dict of IPC socket with each flow, key is flow_id, value is IPCSocket
        self.clients = {}
        self.ipc_file = ipc_file
        self.poll_thread = None

        self.dump_state = dump_state
        # which episode
        self.episode = -1
        # history information of each episode
        self.episode_data = {}
        # this history will be updated on each episode
        self.history = None
        # setup IPC
        self.setup_ipc()

        self.get_action = None
        self.add_to_memory = None

    def setup_ipc(self):
        self.server = IPCSocket()
        self.server.bind(self.ipc_file)
        logger.info("RL master ipc bind on {}".format(self.ipc_file))
        self.server.set_noblocking()
        self.server.listen()
        # register server
        self.poller.add_action(
            Action(
                self.server,
                PollEvents.READ_ERR_FLAGS,
                callback=lambda: self.accept_conn(),
                err_callback=None,
            )
        )

    def run(self):
        self.poll_thread = threading.Thread(target=self.poll)
        self.poll_thread.start()

        # some training semantics
        self.poll_thread.join()

    def poll(self):
        try:
            err_num = 0
            while True:
                if not self.poller.poll_once(-1):
                    # logger.warning("Poller returned false. No need to poll")
                    err_num += 1
                    # if err_num > 5:
                    #     return
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Master run error: {}".format(e))

    def accept_conn(self) -> ReturnStatus:
        # we use poller to add functions
        client = self.server.accept()
        logger.debug("new connection")
        message = client.read()
        message = json.loads(message)
        info = int(message.get("type", -1))
        if info != MessageType.INIT.value:
            raise ValueError(
                "Unknown Message type {}, should be init message".format(info)
            )

        flow_id = int(message["flow_id"])
        logger.info(
            "Master get connection from {}, new ipc fd: {}".format(
                self.ipc_file, client.fileno()
            )
        )

        if flow_id == self.env_flow_id:
            # env IPC
            self.env_ipc = client
            self.poller.add_action(
                Action(
                    client,
                    PollEvents.READ_ERR_FLAGS,
                    callback=lambda: self.read_env_message(client),
                    err_callback=None,
                )
            )
        else:
            # flow IPC
            self.clients[flow_id] = client
            self.poller.add_action(
                Action(
                    client,
                    PollEvents.READ_ERR_FLAGS,
                    callback=lambda: self.read_flow_message(client),
                    err_callback=None,
                )
            )
        return ReturnStatus.Continue

    def read_env_message(self, client: IPCSocket) -> ReturnStatus:
        try:
            raw = client.read(header=True)
            message = json.loads(raw)
            logger.trace("Master received from env: {}".format(message))
            flow_id = int(message["flow_id"])
            if flow_id != self.env_flow_id:
                logger.error(
                    "Unknown sender. Should be {}, actually: {}".format(
                        self.env_flow_id, flow_id
                    )
                )
                return ReturnStatus.Cancel
            logger.trace("Master got message from env: {}".format(message))
            msg_type = int(message.get("type", -1))

            if msg_type == MessageType.START.value:
                # episode start; init history
                self.episode = int(message.get("episode"))
                self.history = History(self.episode, dict(message.get("config")))
                logger.info("Register episode {}".format(self.episode))
            elif msg_type == MessageType.END.value:
                # check episode
                episode = int(message.get("episode"))
                if episode != self.history.episode:
                    raise ValueError(
                        "Episode should be {} rather than {}".format(
                            episode, self.history.episode
                        )
                    )
                self.episode_end(episode)
            elif msg_type == MessageType.ALIVE.value:
                pass
            else:
                raise ValueError("Unknown message type: {}".format(msg_type))
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Client read error: {} from {}".format(e, client.fileno()))
            return ReturnStatus.Cancel
        return ReturnStatus.Continue

    def read_flow_message(self, client: IPCSocket) -> ReturnStatus:
        try:
            raw = client.read(header=True)
            message = json.loads(raw)
            # which flow
            flow_id = int(message["flow_id"])
            # flow send message
            state = dict(message["state"])
            # message type
            status = message.get("type", -1)
            if status == MessageType.OBSERVE.value:
                # observe message
                observer_id = message["observer"]
                step = message["step"]
                logger.trace(
                    "Master received observe message from flow {} for step {} of observer {}".format(
                        flow_id, step, observer_id
                    )
                )
                self.history.add_world_state(observer_id, step, flow_id, state)
                last_state, last_action, last_reward = self.history.get(flow_id, step)
                self.add_to_memory(
                    flow_id, step, last_state, last_action, last_reward, state
                )
                return ReturnStatus.Continue
            elif status == MessageType.END.value:
                logger.debug(
                    "IPC with flow: {} received closing signal".format(flow_id)
                )
                self.clients.pop(flow_id)
                return ReturnStatus.Cancel

        except Exception as e:
            sys.stderr.write(traceback.format_exc())
            logger.error("Client read error: {} from {}".format(e, client.fileno()))
            return ReturnStatus.Cancel

        # normal message for apply RL control
        logger.trace(
            "Master received message from state from flow {} to get action: {}".format(
                flow_id, state
            )
        )
        # first we send signal to observe the world
        step = self.history.add_flow_step(flow_id, state)  # action will be added later
        self.observe_world(flow_id, step)
        # self.pool.apply_async(self.observe_world, args=(flow_id, step))

        # callback will be responsible for writing action back to flow
        # encapsulate callback
        whole_callback = partial(
            self.write_action, client=client, flow_id=flow_id, step=step
        )

        self.pool.apply_async(
            self.sample_action, args=(flow_id, state), callback=whole_callback
        )
        return ReturnStatus.Continue

    def observe_world(self, observer_id, step):
        # if only contains one network flow, it must be the observer itself
        if len(self.clients) == 1:
            return
        for flow_id, client in self.clients.items():
            if flow_id != observer_id:
                message = {}
                message["type"] = MessageType.OBSERVE.value
                message["observer"] = observer_id
                message["step"] = step
                client.write(json.dumps(message))
                logger.trace(
                    "Master send observe signal to flow {} for observer: {}".format(
                        flow_id, observer_id
                    )
                )

    def write_action(self, action, client, flow_id, step):
        """Write action back to each learning flow

        `action` must be the first argument to enable function wrapper of partial

        Args:
            action (int): cwnd adjustment
            client (IPCSocket): which client to write
            flow_id (flow id): which flow to write to
        """
        reply = {}
        reply["flow_id"] = flow_id
        reply["type"] = MessageType.ALIVE.value
        reply["cwnd"] = action
        logger.trace("Master will write back: {} for flow {}".format(action, flow_id))
        client.write(json.dumps(reply))
        # save action to flow_state, state has been included
        self.history.add_flow_step_action(flow_id, step, action)

    def sample_action(self, flow_id, state):
        """This function provides inference service.

        This function is compute-intensive, so we will apply async on it
        and make the writing back as callback.

        Args:
            flow_id (int): flow id
            state (dict): current state information

        Returns:
            int: control action
        """
        # action = self.agents(flow_id, state)
        # emulate the computation time: 5ms
        # time.sleep(0.003)
        action = 400
        # self.write_action(action, self.clients[flow_id], flow_id)
        return action

    def episode_end(self, episode):
        logger.info("ENV ends episode {}".format(episode))
        # save history
        self.episode_data[self.episode] = copy.deepcopy(self.history)
        if self.dump_state:
            # self.pool.apply_async(self.dump_data, args=(self.history))
            self.dump_data(self.history)
        # there may need some other operations, such as traning
        return

    def add_to_history(self, flow_id, state, reward, action):
        pass

    def calculate_reward(self, flow_id, state, action):
        return 1

    def dump_data(self, data, file_path="data.json"):
        logger.info("Dump history of episode {} to file".format(self.episode))
        with open(file_path, "a") as f:
            json.dump(data.to_json(), f, cls=ComplexEncoder)

    @staticmethod
    def make_master(self, num_flow, ipc_file):
        self.__init__(ipc_file, num_flow)
        return self
