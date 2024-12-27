import ipaddress
import os
import sys
import struct

from logger import logger as log

# message types
NL_CREATE = 0
NL_MEASURE = 1
NL_INSTALL_EXPR = 2
NL_UPDATE_FIELDS = 3
NL_CHANGE_PROG = 4
NL_READY = 5
# spine message types
NL_STATE = 6
NL_PARAM = 7
NL_NEURAL_NETWORK = 8
NL_RELEASE = 9

# type of registers
VOLATILE_CONTROL_REG = 8

# registers for sCubic Parameters
CUBIC_BIC_SCALE_REG = 0
CUBIC_BETA_REG = 1

# registers for Vanilla Parameters
VANILLA_ALPHA_REG = 0
VANILLA_BETA_REG = 1
VANILLA_GAMMA_REG = 2
VANILLA_DELTA_REG = 3

# registers for Neo Parameters
NEO_ACTION_REG = 0

# some length of message
SPINE_HEADER_LEN = 8
SPINE_CREATE_LEN = 88

# read u32 from big endian
# raw: AB CD EF GH -> Big endian: GH EF CD AB
# mid little endian: EF GH AB CD


def u64_from_bytes(bytes):
    # Full big endian:
    tmp = struct.unpack("<I", bytes)[0]
    b1 = tmp & 0xFF00000000000000
    b2 = tmp & 0x00FF000000000000
    b3 = tmp & 0x0000FF0000000000
    b4 = tmp & 0x000000FF00000000
    b5 = tmp & 0x00000000FF000000
    b6 = tmp & 0x0000000000FF0000
    b7 = tmp & 0x000000000000FF00
    b8 = tmp & 0x00000000000000FF
    return (
        (b1 >> 8)
        + (b2 << 8)
        + (b3 >> 8)
        + (b4 << 8)
        + (b5 >> 8)
        + (b6 << 8)
        + (b7 >> 8)
        + (b8 << 8)
    )


def u32_from_bytes(bytes):
    # Full big endian:
    tmp = struct.unpack("<I", bytes)[0]
    b1 = tmp & 0xFF000000  # GH
    b2 = tmp & 0x00FF0000  # EF
    b3 = tmp & 0x0000FF00  # CD
    b4 = tmp & 0x000000FF  # AB
    return (b1 >> 8) + (b2 << 8) + (b3 >> 8) + (b4 << 8)


def u16_from_bytes(bytes):
    # Full big endian:
    tmp = struct.unpack("<H", bytes)[0]

    b1 = tmp & 0xFF000000  # GH
    b2 = tmp & 0x00FF0000  # EF
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


# this class deals with incoming measurment messages from kernel, rather than the outgoing one
class MeasureMsg(object):
    def __init__(self):
        self.msg_fields_len = 4 * 2
        self.int_raw_format = "<II"
        self.init_len = struct.calcsize(self.int_raw_format)

        self.request_id = 0
        self.field_num = 0
        self.data = []

    def from_raw(self, buf):
        # first process message
        if len(buf) < self.init_len:
            log.error("message length too small")
            return False
        self.request_id = struct.unpack("<I", buf[0:4])[0]
        self.field_num = struct.unpack("<I", buf[4:8])[0]
        
        if len(buf) < self.init_len + int(self.field_num) * 8:
            log.error("incomplete state message")
            return False

        for i in range(self.field_num):
            self.data.append(
                struct.unpack("<Q", buf[self.init_len + i * 8 : self.init_len + (i + 1) * 8])[0]
            )
        return True

# this class deals with outgoing measurement message to kernel that requests states, not the incoming one
class MeasureRequestMsg(object):
    def __init__(self, request_id):
        self.request_id = request_id

    def serialize(self):
        buf = struct.pack("<I", self.request_id)
        return buf


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
