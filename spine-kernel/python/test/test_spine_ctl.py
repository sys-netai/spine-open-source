import os
from pickle import OBJ
import sys
import json
import time
import argparse
import threading
import functools
from functools import partial

import context
from logger import logger as log
from message import *
from netlink import Netlink
from ipc_socket import IPCSocket
from spine_flow import ActiveFlowMap, Flow
from poller import Action, Poller, ReturnStatus, PollEvents

# communication between spine-user-space and kernel
nl_sock = None
# communication between spine-user-space and Env
unix_sock = None
# cont status of polling
cont = threading.Event()
poller = Poller()

cubic_beta = 717
cubic_bic_scale = 41
step = 10

active_flowmap = ActiveFlowMap()


def build_unix_sock(unix_file):
    sock = IPCSocket()
    sock.connect(unix_file)
    sock.set_noblocking()
    return sock


def build_netlink_sock():
    sock = Netlink()
    sock.add_mc_group()
    return sock


def read_netlink_message(nl_sock: Netlink):
    hdr_raw = nl_sock.next_msg()
    if hdr_raw == None:
        return ReturnStatus.Cancel
    hdr = SpineMsgHeader()
    if hdr.from_raw(hdr_raw) == None:
        log.error("Failed to parse netlink header")
    log.info("recv netlink message: {}".format(hdr.type))
    if hdr.type == CREATE:
        msg = CreateMsg()
        msg.from_raw(hdr_raw[hdr.hdr_len :])
        flow = Flow().from_create_msg(msg, hdr)
        if active_flowmap.add_flow_with_sockId(flow):
            global cubic_bic_scale
            global cubic_beta
            cubic_beta += step
            cubic_bic_scale += step
            log.info("send control message: {}".format(flow.sock_id))
            send_control_message(flow.sock_id, cubic_beta, cubic_bic_scale)
            time.sleep(3)
            cubic_beta += step
            cubic_bic_scale += step
            log.info("send control message: {}".format(flow.sock_id))
            send_control_message(flow.sock_id, cubic_beta, cubic_bic_scale)
        return ReturnStatus.Continue
    elif hdr.type == READY:
        log.info("Spine kernel is ready!")
    elif hdr.type == MEASURE:
        sock_id = hdr.sock_id
        active_flowmap.remove_flow_by_sockId(sock_id)

def send_control_message(flow_id, cubic_beta, cubic_bic_scale):
    msg = UpdateMsg()
    # first bic_scale, then beta
    msg.add_field(
        UpdateField().create(VOLATILE_CONTROL_REG, CUBIC_BIC_SCALE_REG, cubic_bic_scale)
    )
    msg.add_field(
        UpdateField().create(VOLATILE_CONTROL_REG, CUBIC_BETA_REG, cubic_beta)
    )
    update_msg = msg.serialize()
    nl_hdr = SpineMsgHeader()
    # sock_id = flow_id_map[flow_id]
    sock_id = flow_id
    nl_hdr.create(UPDATE_FIELDS, len(update_msg) + nl_hdr.hdr_len, sock_id)
    nl_sock.send_msg(nl_hdr.serialize() + update_msg)


def polling():
    try:
        while not cont.is_set():
            poller.poll_once()
    except KeyboardInterrupt:
        sys.exit(1)


def main(args):
    # recv new spine flow info and misc
    netlink_read_wrapper = partial(read_netlink_message, nl_sock)
    poller.add_action(
        Action(nl_sock, PollEvents.READ_ERR_FLAGS, callback=netlink_read_wrapper)
    )
    threading.Thread(target=polling).run()
    # time.sleep(3)
    # while True:
    #     if len(active_flowmap) != 0:
    #         for sock_id in active_flowmap.keys():
    #             break
    #     time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ipc", "-i", type=str, required=True, help="IPC communication between Env and Spine controller")
    args = parser.parse_args()
    # build communication sockets
    nl_sock = build_netlink_sock()

    main(args)
