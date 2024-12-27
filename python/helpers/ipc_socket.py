import os
import sys
import socket
import struct
import errno
from traceback import print_exception
import traceback
import time
import queue
from queue import Queue

from . import context
from helpers.logger import logger as logger


class IPCSocket(object):
    def __init__(self):
        self.__sock: socket.socket = socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM, 0
        )
        self.connected = False
        self._set_reuseaddr()
        # message buffer
        self.message_buffer = list()
        # length of all buffered bytes
        self.buffer_length = 0
        # write offset
        self.offset = 0

    def bind(self, ipc_file):
        self.ipc_file = ipc_file
        self.__sock.bind(ipc_file)
        logger.info("IPCSocket bind on {}".format(ipc_file))

    def listen(self, backlog=16):
        self.__sock.listen(backlog)

    @staticmethod
    def from_socket(sock: socket.socket):
        ipc = IPCSocket()
        ipc.__sock = sock
        ipc.connected = True
        return ipc

    def accept(self):
        client, _ = self.__sock.accept()
        return IPCSocket.from_socket(client)

    def settimeout(self, *args):
        return self.__sock.settimeout(*args)

    def connect(self, ipc_file):
        self.__sock.connect(ipc_file)
        self.connected = True
        return self

    def read(self, header=True, header_len=2, count=1024):
        if not self.connected:
            logger.error("Socket not connected")
            return None
        if header:
            # header length in unsigned short
            raw = self.__sock.recv(header_len, socket.MSG_WAITALL)
            data_len = 0
            if len(raw) == header_len:
                data_len = struct.unpack("!H", raw)[0]
                # read all data
                read_len = 0
                data_recv = []
                while read_len < data_len:
                    try:
                        data = self.__sock.recv(data_len - read_len)
                        if data == b"":
                            logger.warning(
                                "IPCSocket: read 0 data from {}".format(self.fileno())
                            )
                            self.close()
                            return None
                        else:
                            data_recv.append(data)
                            read_len += len(data)
                    except socket.error as e:
                        print_exception(e)
                        logger.warning(
                            "IPC socket read error (expected: {}, actually: {}), close it".format(
                                data_len - read_len, len(data)
                            )
                        )
                        self.close()
                    except socket.timeout as e:
                        print_exception(e)
                        logger.error("IPCSocket: read timeout")
                        break
                return b"".join(data_recv)
            else:
                # there may be some error
                logger.warning(
                    "IPC socket read error (expected: {}, actually: {}), close it".format(
                        header_len, data_len
                    )
                )
                self.close()
                return None
        else:
            return self.__sock.recv(count)

    def add_to_buffer(self, message: str, prepend_message_len=True):
        message = message.encode("utf-8")
        if prepend_message_len:
            head = len(message)
            # header_length in unsigned short
            header = struct.pack("!H", head)
            message = header + message
        self.message_buffer.append(message)
        self.buffer_length += len(message)
        return self.buffer_length

    def write_once(self):
        """Dump message from message buffer to IPC"""
        if not self.connected:
            return -1

        if len(self.message_buffer) == 0:
            return 0

        message: str = self.message_buffer[0]
        send_cnt = -1
        # write from the last_write_offset of buffer
        try:
            send_cnt = self.__sock.send(message[self.offset :])
        except OSError as e:
            sys.stderr.write(traceback.format_exc())
            logger.error("To write {}".format(message))
            if e.errno == errno.EPIPE:
                logger.error(
                    "IPCSocket: Catch EPIPE error on fd {}. Close the IPC".format(
                        self.fileno()
                    )
                )
                self.close()
                return -1
            elif e.errno == errno.EAGAIN:
                logger.warning(
                    "IPCSocket: Catch EAGAIN warning on fd {}. Try again".format(
                        self.fileno()
                    )
                )
                # we may continue next time
                return 0
            elif e.errno == errno.EBADF:
                logger.error("IPCSocket: bad filedescriptor: {}".format(self.fileno()))
                self.close()
                return -1
            else:
                logger.warning(
                    "IPCSocket: write error {}; write_cnt: {}".format(e, send_cnt)
                )
                return 0
        # logger.trace(
        #     "IPCSocket: write to {}, total message len: {}, send size: {}, write: {}".format(
        #         self.fileno(),
        #         len(message),
        #         send_cnt,
        #         message[self.offset : self.offset + send_cnt],
        #     )
        # )
        # update offset
        self.offset += send_cnt
        if self.offset == len(message):
            # remove the first element
            self.message_buffer.pop(0)
            self.offset = 0
        # update total buffer size
        self.buffer_length -= send_cnt
        return send_cnt

    def write(self, message: str, prepend_message_len=True):
        if not self.connected:
            # logger.error("Socket not connected")
            return -1
        message = message.encode("utf-8")
        if prepend_message_len:
            head = len(message)
            # header_length in unsigned short
            header = struct.pack("!H", head)
            message = header + message

        message_len = len(message)
        # self.__sock.sendall(message) cannot give sent count if error occurred

        # we attend to write all
        sent_len = 0
        while sent_len < message_len:
            send_cnt = 0
            try:
                send_cnt = self.__sock.send(message)
            except OSError as e:
                sys.stderr.write(traceback.format_exc())
                logger.error("To write {}".format(message))
                if e.errno == errno.EPIPE:
                    logger.error(
                        "IPCSocket: Catch EPIPE error on fd {}. Close the IPC".format(
                            self.fileno()
                        )
                    )
                    self.close()
                    return -1
                elif e.errno == errno.EAGAIN:
                    logger.warning(
                        "IPCSocket: Catch EAGAIN warning on fd {}. Try again".format(
                            self.fileno()
                        )
                    )
                    time.sleep(10 / 1000)
                    continue
                elif e.errno == errno.ETIMEDOUT:
                    logger.warning(
                        "IPCSocket: write timeout on {}".format(self.fileno())
                    )
                    time.sleep(10 / 1000)
                    continue
                else:
                    self.close()
                    return -1
            sent_len += send_cnt
        return sent_len

    def _set_reuseaddr(self):
        self.__sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def get_origin_socket(self):
        return self.__sock

    def fileno(self) -> int:
        return self.__sock.fileno()

    def close(self):
        if self.connected:
            logger.info("IPCSocket: Close the IPC")
            self.connected = False
            return self.__sock.close()

    def set_noblocking(self):
        self.__sock.setblocking(0)
