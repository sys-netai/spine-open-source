import ipaddress
import os
import sys
import struct

from logger import logger as log

# message types
CREATE = 0
MEASURE = 1
INSTALL_EXPR = 2
UPDATE_FIELDS = 3
CHANGE_PROG = 4
READY = 5
# spine message types
STATE = 6
PARAM = 7
NEURAL_NETWORK = 8

# type of registers
VOLATILE_CONTROL_REG = 8

# registers for Cubic Parameters
CUBIC_BIC_SCALE_REG = 0
CUBIC_BETA_REG = 1

# some length of message
SPINE_HEADER_LEN = 8
SPINE_CREATE_LEN = 88

# read u32 from big endian
# raw: AB CD EF GH -> Big endian: GH EF CD AB
# mid little endian: EF GH AB CD 
def u32_from_bytes(bytes):
    # Full big endian: 
    tmp = struct.unpack("<I", bytes)[0]

    b1 = tmp & 0xff000000 # GH
    b2 = tmp & 0x00ff0000 # EF
    b3 = tmp & 0x0000ff00 # CD
    b4 = tmp & 0x000000ff # AB 
    return (b1 >> 8) + (b2 << 8) + (b3 >> 8) + (b4 << 8)

def u16_from_bytes(bytes):
    # Full big endian: 
    tmp = struct.unpack("<H", bytes)[0]

    b1 = tmp & 0xff000000 # GH
    b2 = tmp & 0x00ff0000 # EF
    return (b1 >> 8) + (b2 << 8)



class SpineMsgHeader(object):
    def __init__(self):
        self.hdr_len = 2 + 2 + 4
        # u16
        self.type = -1
        # u16
        self.len = -1
        # u32
        self.sock_id = -1
        self.raw_format = "=HHI"

    def from_raw(self, buf):
        if not isinstance(buf, bytes):
            log.error("expected bytes")
            return None
        if len(buf) < self.hdr_len:
            log.error("header length too small")
            return None
        self.type, self.len, self.sock_id = struct.unpack(self.raw_format, buf[0:8])
        return self

    def create(self, type, len, sock_id):
        self.type = type
        self.len = len
        self.sock_id = sock_id

    def serialize(self):
        return struct.pack(self.raw_format, self.type, self.len, self.sock_id)


class CreateMsg(object):
    def __init__(self):
        self.msg_len = 4 * 6 + 64
        self.int_raw_format = "<IIIIII"
        self.int_len = struct.calcsize(self.int_raw_format)

        self.init_cwnd = 0
        self.mss = 0
        self.src_ip = 0
        self.src_port = 0
        self.dst_ip = 0
        self.dst_port = 0
        # max 64 bytes
        self.congAlg = ""

    def from_raw(self, buf):
        # first process message
        if len(buf) < self.msg_len:
            log.error("message length too small")
        # print("init_cwnd buf in hex: {}".format(buf[12:16]))
        self.init_cwnd = u32_from_bytes(buf[0:4])
        self.mss = u32_from_bytes(buf[4:8])
        # print("ip raw: {}".format(buf[8:12]))
        self.src_ip = struct.unpack("!I", buf[8:12])[0]
        # self.src_ip = struct.unpack("<I", buf[8:12])[0]
        self.src_port = int(u32_from_bytes(buf[12:16]))
        self.dst_ip = struct.unpack("!I", buf[16:20])[0]
        self.dst_port = int(u32_from_bytes(buf[20:24]))
        # remaining part is char array
        self.congAlg = buf[self.int_len :].decode()
        return self


class UpdateField(object):
    def __init__(self):
        self.field_len = 1 + 4 + 8
        self.raw_format = "<BIQ"

        self.reg_type = -1
        self.reg_index = -1
        self.new_value = -1

    def create(self, type, index, value):
        self.reg_type = type
        self.reg_index = index
        self.new_value = value
        return self

    def serialize(self):
        return struct.pack(
            self.raw_format, self.reg_type, self.reg_index, self.new_value
        )

    def deserialize(self, buf):
        if len(buf) < self.field_len:
            log.error("message length too small")
        self.reg_type, self.reg_index, self.new_value = struct.unpack(
            self.raw_format, buf[: self.field_len]
        )


class UpdateMsg(object):
    def __init__(self):
        self.num_fields = 0
        self.fields = []

    def add_field(self, field):
        self.fields.append(field)
        self.num_fields += 1

    def serialize(self):
        buf = struct.pack("<I", self.num_fields)
        for field in self.fields:
            buf += field.serialize()
        return buf

    def deserialize(self, buf):
        self.num_fields = struct.unpack("<I", buf[0:4])
        for i in range(self.num_fields):
            field = UpdateField()
            field.deserialize(buf[4 + i * field.field_len :])
            self.fields.append(field)
        return self


def ReadyMsg(object):
    def __init__(self):
        self.msg_len = 4
        # u32
        self.ready = 0

    def serialize(self):
        return struct.pack("<I", self.ready)

    def deserialize(self, buf):
        self.ready = struct.unpack("<I", buf[0:4])
