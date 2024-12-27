from enum import Enum
import os
import stat
import sys
import json
import time
import argparse
import threading
import functools
from functools import partial


from logger import logger as log
from message import *
from netlink import Netlink
from ipc_socket import IPCSocket
from spine_flow import Flow, ActiveFlowMap, EnvFlows
from poller import Action, Poller, ReturnStatus, PollEvents
from helper import drop_privileges
import msg_sender

# log.setLevel("INFO")
# communication between spine-user-space and kernel
nl_sock = None
# communication between spine-user-space and Env
unix_sock = None
client_from_sock = dict()
# kernel cc alg
kernel_cc = None
# message sender
nl_send = None
# cont status of polling
cont = threading.Event()
poller = Poller()

# we do require one active flow map here
env_flows = EnvFlows()
env_id = "spine_eval"
global_flow_id = 0


class UnixMessageType(Enum):
    INIT = 0  # env initialization
    START = 1  # episode start
    END = 2  # episode end
    ALIVE = 3  # alive status
    OBSERVE = 4  # observe the world
    TERMINATE = 5  # terminate the env\\
    MEASURE = 6


def build_unix_sock(unix_file):
    if os.path.exists(unix_file):
        log.debug("{} already exists, remove it".format(unix_file))
        os.remove(unix_file)
    sock = IPCSocket()
    log.debug("UNIX IPC file: {}".format(unix_file))
    sock.bind(unix_file)
    sock.set_noblocking()
    sock.listen()
    log.info("Spine is listening for flows from env at {}".format(unix_file))
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
        return ReturnStatus.Cancel
    if hdr.type == NL_CREATE:
        msg = CreateMsg()
        msg.from_raw(hdr_raw[hdr.hdr_len :])
        flow = Flow().from_create_msg(msg, hdr)
        # first find env
        env_id = env_flows.dst_port_to_env_id.get(flow.dst_port, None)
        if env_id == None:
            log.warn("unknown dst_port: {}".format(flow.dst_port))
            return ReturnStatus.Continue
        # register new flow
        active_flow_map = env_flows.get_env_flows(env_id)
        if active_flow_map == None:
            log.warn("env: {} has not registered".format(env_id))
            return ReturnStatus.Continue
        active_flow_map.add_flow_with_sockId(flow)
        # cache sockID with envid
        env_flows.bind_sock_id_to_env(flow.sock_id, env_id)
        return ReturnStatus.Continue
    elif hdr.type == NL_READY:
        log.info("Spine kernel is ready!!")
    elif hdr.type == NL_MEASURE:
        sock_id = hdr.sock_id
        msg = MeasureMsg()
        msg.from_raw(hdr_raw[hdr.hdr_len :])
        msg_to_unix = {}
        msg_to_unix["type"] = UnixMessageType.MEASURE.value
        msg_to_unix["data"] = msg.data
        msg_to_unix["request_id"] = msg.request_id
        client_sock = client_from_sock[sock_id]
        sent = client_sock.write(json.dumps(msg_to_unix))
        if sent == 0:
            print("Nothing sent...")
        
    elif hdr.type == NL_RELEASE:        
        # flow release
        sock_id = hdr.sock_id
        # we just remove the cached items
        env_flows.release_sock_id_to_env(sock_id)
        # env has been deregistered, do nothing
    return ReturnStatus.Continue


def make_flow_id(port):
    return hash(port)


def read_unix_message(unix_sock: IPCSocket):
    raw = unix_sock.read(header=True)
    if raw == None:
        return ReturnStatus.Cancel
    data = json.loads(raw)
    # env_id = str(data["env_id"])
    port = int(data["dst_port"])
    msg_type = data["type"]

    active_flow_map = env_flows.get_env_flows(env_id)
    if active_flow_map is None:
        return ReturnStatus.Cancel

    if msg_type == UnixMessageType.START.value:
        # we also need to record the corresponce of env_id and dst_port
        flow_id = make_flow_id(port)
        env_flows.bind_port_to_env(port, env_id)
        active_flow_map.add_flow_with_dst_port(port, flow_id)
        log.info("new flow with port: {}, assigned flow id: {}".format(port, flow_id)) 
        return ReturnStatus.Continue

    flow_id = active_flow_map.get_flowId_by_port(port)
    assert flow_id != None

    if msg_type == UnixMessageType.END.value:
        # we need the dsr_port id to remove the cache
        sock_id = active_flow_map.get_sockId_by_flowId(flow_id)
        log.info("flow exits with port: {}".format(port)) 
        port = active_flow_map.remove_flow_by_flowId(flow_id)
        # remove cached items
        env_flows.release_port_to_env(port)
        return ReturnStatus.Cancel
    
    elif msg_type == UnixMessageType.MEASURE.value:
        # we need the dsr_port id to remove the cache
        sock_id = active_flow_map.get_sockId_by_flowId(flow_id)
        client_from_sock[sock_id] = unix_sock # cache the client socket
        assert nl_send != None
        if data["request_id"] is None:
            return ReturnStatus.Continue
        
        if sock_id == None:
            print("Warning: a none sock_id")
            return ReturnStatus.Continue
        nl_send(data["request_id"], nl_sock, sock_id, msg_type=NL_MEASURE)
        return ReturnStatus.Continue
    # message should be ALIVE
    if msg_type != UnixMessageType.ALIVE.value:
        log.error("Incorrect message type: {}".format(msg_type))
        return ReturnStatus.Cancel
    # lookup sock id by flow_id
    sock_id = active_flow_map.get_sockId_by_flowId(flow_id)
    if sock_id == None:
        log.warn("unknown flow id: {}".format(flow_id))
        return ReturnStatus.Cancel

    # spine semantics: None means no action is need
    if data["action"] is None:
        return ReturnStatus.Continue

    assert nl_send != None
    nl_send(data["action"], nl_sock, sock_id)
    return ReturnStatus.Continue


def accept_unix_conn(unix_sock: IPCSocket, poller: Poller):
    client: IPCSocket = unix_sock.accept()
    # deal with init message
    # message = client.read()
    # message = json.loads(message)
    # info = int(message.get("type", -1))
    # in eval mode, each flow directly connects spine with START message
    # if info != UnixMessageType.START.value:
    #     log.error("Incorrect message type: {}, ignore this".format(info))
    #     return ReturnStatus.Continue
    # accept new conn and register to poller
    # env_id = str(message["env_id"])
    log.debug(
        "Spine registers flow, env_id is {}, new ipc fd: {}".format(
            env_id, client.fileno()
        )
    )
    
    client.set_noblocking()
    # unix_sock: recv updated parameters and relay to nl_sock
    unix_read_wrapper = partial(read_unix_message, client)
    poller.add_action(
        Action(client, PollEvents.READ_ERR_FLAGS, callback=unix_read_wrapper)
    )
    return ReturnStatus.Continue


def polling():
    while not cont.is_set():
        if poller.poll_once() == False:
            # just sleep for a while (10ms)
            
            #print("CURRENT POLLER ACTION in polling:", poller.get_all_actions())
            time.sleep(0.01)


def main(args):
    # register accept for unix socket
    listen_callback = partial(accept_unix_conn, unix_sock, poller)
    poller.add_action(
        Action(
            unix_sock,
            PollEvents.READ_ERR_FLAGS,
            callback=listen_callback,
        )
    )
    # create the default env for spine
    env_flows.register_env(env_id)

    # recv new spine flow info and misc
    netlink_read_wrapper = partial(read_netlink_message, nl_sock)
    poller.add_action(
        Action(nl_sock, PollEvents.READ_ERR_FLAGS, callback=netlink_read_wrapper)
    )
    # print("CURRENT POLLER ACTION:", poller.get_all_actions())
    threading.Thread(target=polling).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_unix_file = "/tmp/spine_ipc"
    parser.add_argument(
        "--ipc",
        "-i",
        type=str,
        default=None,
        help="IPC communication between Env and Spine controller",
    )
    parser.add_argument(
        "--user",
        "-u",
        type=str,
        default="tianhan",
        help="the effective user after drop root privileges, we assume the same gid_name as uid_name",
    )
    parser.add_argument(
        "--alg",
        "-a",
        type=str,
        default="vanilla",
        help="kernel CC algorithm",
    )
    args = parser.parse_args()
    nl_sock = build_netlink_sock()
    drop_privileges(uid_name=args.user, gid_name=args.user)
    # after build netlink socket, we try to drop root privilege
    # build communication sockets
    if args.ipc is None:
        args.ipc = "{}".format(default_unix_file)
    unix_sock = build_unix_sock(args.ipc)
    # mod: 775
    os.chmod(args.ipc, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
    # assign netlink message sender
    kernel_cc = args.alg
    nl_send = getattr(msg_sender, "send_{}_message".format(kernel_cc)) 
    main(args)
