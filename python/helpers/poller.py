import enum
import os
import sys
import select
import socket
import traceback
from typing import NamedTuple
from enum import Enum

from . import context
from helpers.ipc_socket import IPCSocket
from helpers.logger import logger as logger


class PollEvents(Enum):
    READ_FLAGS = select.POLLIN | select.POLLPRI
    WRITE_FLAGS = select.POLLOUT
    ERR_FLAGS = select.POLLERR | select.POLLHUP | select.POLLNVAL
    READ_ERR_FLAGS = READ_FLAGS | ERR_FLAGS
    ALL_FLAGS = READ_FLAGS | WRITE_FLAGS | ERR_FLAGS

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ReturnStatus(Enum):
    Continue = 0
    Cancel = 1


class Action(object):
    def __init__(self, sock, event: PollEvents, callback, err_callback=None):
        if not (isinstance(sock, socket.socket) or isinstance(sock, IPCSocket)):
            raise TypeError(
                "sock should be socket.socket or IPCSocket rather than {}".format(
                    type(sock)
                )
            )
        self.sock = sock
        # check event
        if not PollEvents.has_value(event.value):
            raise ValueError("Poller: invalid eventmask: {}".format(event))
        self.event: PollEvents = event
        self.callback = callback
        self.err_callback = err_callback
        self.fd = self.sock.fileno()


class Poller(object):
    def __init__(self):
        self.__poller = select.poll()
        self.__action_to_add = []
        # avoid redundant elements
        self.__fd_to_remove = set()
        self.__actions = []
        # avoid repeatedly register, key is fd, value is mask
        self.registered_fd = {}

    def add_action(self, action: Action):
        # first add to tmp queue
        self.__action_to_add.append(action)

    def remove_fd(self, fd: int):
        # tmp queue to store fd to be removed
        self.__fd_to_remove.add(fd)

    def __remove_action(self):
        if len(self.__fd_to_remove) == 0:
            return

        for fd in self.__fd_to_remove:
            if fd not in self.registered_fd:
                logger.error("Poller: FileDescriptor {} is not registered".format(fd))
                break
            # multiple actions for one fd
            for action in reversed(self.__actions):
                if action.fd == fd:
                    # one fd may have multiple actions
                    if fd in self.registered_fd:
                        self.unregister(fd)
                        self.registered_fd.pop(fd)
                        logger.debug(
                            "Poller: Remove fd: {} from registered fd list".format(fd)
                        )
                    self.__actions.remove(action)
                    logger.debug(
                        "Poller: Removed action and unregister fd: {}, remove mask of {}, current action_len: {}".format(
                            action.fd, action.event.name, len(self.__actions)
                        )
                    )
        logger.debug(
            "Poller: finish removing action, current action_len is {}".format(
                len(self.__actions)
            )
        )

    def get_action_by_fd(self, fd):
        ret_actions = []
        for action in self.__actions:
            if action.fd == fd:
                ret_actions.append(action)
        return ret_actions

    def poll_once(self, timeout=1000) -> bool:
        # add newly arrived fd
        if len(self.__action_to_add) != 0:
            self.__actions += self.__action_to_add
            self.__action_to_add = []

        # logger.debug("Poller: start Poll, action len: {}".format(len(self.__actions)))
        if len(self.__actions) == 0:
            # logger.warning("No valid action to poll")
            return False

        # add poll target
        for action in self.__actions:
            # if not registered, then register
            if action.fd not in self.registered_fd:
                self.__poller.register(action.fd, action.event.value)
                self.registered_fd[action.fd] = action.event.value
                logger.debug(
                    "Poller: add fd: {} with event: {}(mask: {})".format(
                        action.fd, action.event.name, action.event.value
                    )
                )
            # we need to set new bit in event mask
            elif action.event.value & self.registered_fd[action.fd] == 0:
                current_mask = self.registered_fd[action.fd]
                newmask = current_mask | action.event.value
                self.__poller.modify(action.fd, newmask)
                self.registered_fd[action.fd] = newmask
                logger.debug(
                    "Poller: modified eventmask of fd: {} from {} to be {}. Add: {}".format(
                        action.fd, current_mask, newmask, action.event.name
                    )
                )

        # logger.debug("Poller: registered action")
        while True:
            try:
                events = self.__poller.poll(timeout)
                break
            except InterruptedError as e:
                logger.error("Poller: InterruptedError - {}".format(e))
                continue
        # logger.debug("Poller: end Poll")
        if not events:
            logger.warning("Poller: No event in polling")
            return False

        # logger.debug("Poller: have action {}".format(len(events)))
        for fd, flag in sorted(events):
            actions = self.get_action_by_fd(fd)
            if flag & PollEvents.ERR_FLAGS.value:
                # we select the first action to run error callback
                action = actions[0]
                logger.error(
                    "Poller: Error on poll fd {}, error is {}".format(
                        fd, flag & PollEvents.ERR_FLAGS.value
                    )
                )
                logger.debug(
                    "Poller, to run the error callback of fileno: {}".format(
                        action.sock.fileno()
                    )
                )
                if action.err_callback:
                    try:
                        action.err_callback()
                    except Exception as e:
                        sys.stderr.write(traceback.format_exc())
                        logger.error("Poller: error in err_callback: {}".format(e))
                self.remove_fd(fd)
                continue
                # continue to serve other fds
            # one fd may relate to multiple actions, e.g., read or write
            for action in actions:
                if flag & action.event.value:
                    # logger.debug("Poller: fd {} is readable".format(fd))
                    try:
                        # logger.debug(
                        #     "Poller: begin to run callback on fd {}".format(fd)
                        # )
                        status = action.callback()
                        if status == ReturnStatus.Continue:
                            # all is normal, do nothing
                            continue
                        elif status == ReturnStatus.Cancel:
                            logger.info("Poller: Cancel polling on fd: {}".format(fd))
                            self.remove_fd(fd)
                            break
                    except Exception as e:
                        sys.stderr.write(traceback.format_exc())
                        logger.error("Poller: error in callback: {}".format(e))
                        self.remove_fd(fd)
                        # break loop on actions
                        break
        # logger.debug("Poller: remove action")
        self.__remove_action()
        self.__fd_to_remove.clear()
        return True

    def clear_all(self):
        for index, action in enumerate(self.__actions.copy()):
            self.__actions.pop(index)

    def unregister(self, fd):
        try:
            self.__poller.unregister(fd)
        except KeyError as e:
            logger.warning("Poller: fd {} is not registered in poller".format(fd))
            return
