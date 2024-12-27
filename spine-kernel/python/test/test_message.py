import unittest

import context
from message import *


class TestMessage(unittest.TestCase):
    def test_spine_msg_header(self):
        hdr = SpineMsgHeader()
        hdr.type = CREATE      # 0x0000
        hdr.len = 101          # 0x0065
        hdr.sock_id = 123456   # 0x0001E240
        raw = hdr.serialize()
        print(" ".join(["{:02x}".format(x) for x in raw]))
        self.assertEqual(raw, b'\x00\x00\x65\x00\x40\xE2\x01\00')