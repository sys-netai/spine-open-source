import socket
import os
import sys
import struct

import context
from logger import logger as log 

SOL_NETLINK = 270
NETLINK_ADD_MEMBERSHIP = 1

NETLINK_GROUP = 22

MAX_PAYLOAD_LEN = 1024


class Netlink(object):
    def __init__(self, protocol=socket.NETLINK_USERSOCK, nl_group=0):
        self.sock = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, protocol)
        assert self.sock.fileno() > 0, "Invalid socket fileno"
        # bind <port, group>
        # set port to 0 to let kernel assign port number (usually the pid of current process)
        # The default value for this field (group)is zero which means that no multicasts will be received
        self.sock.bind((0, nl_group))
        self.port = self.sock.getsockname()[0]
        self.nl_header_len = 16

    def fileno(self):
        return self.sock.fileno()

    def add_mc_group(self, group=NETLINK_GROUP):
        self.sock.setsockopt(SOL_NETLINK, NETLINK_ADD_MEMBERSHIP, group)

    def send_msg(self, message: bytes, des=0):
        total_len = len(message) + self.nl_header_len
        if total_len > MAX_PAYLOAD_LEN:
            sys.stderr.write("netlink message is too long.")
            return -1
        buf = self.write_header(total_len) + message
        # des 0 means sending to kernel
        # we don't need to specify nl_group here
        return self.sock.sendto(buf, (des, 0))

    def recv_raw(self, buf_size=1024):
        return self.sock.recv(buf_size)

    def write_header(self, len=MAX_PAYLOAD_LEN):
        """prepare netlink header (16 bytes)
        0               1               2               3
        0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                          Length                             |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |            Type              |           Flags              |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                      Sequence Number                        |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                      Process ID (PID)                       |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        """
        nl_header = struct.pack("=IHHII", len, 0, 0, 0, self.port)
        return nl_header

    def next_msg(self):
        buf = self.recv_raw()
        if len(buf) <= self.nl_header_len:
            log.warn("netlink message is too short. May no real payload")
            return None
        return buf[self.nl_header_len:]
        
        
