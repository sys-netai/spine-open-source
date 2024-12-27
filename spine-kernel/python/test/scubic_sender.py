import os
import sys
import socket
from random import choice
from string import ascii_uppercase

import context
from logger import logger as log

def random_string(len):
    return ''.join(choice(ascii_uppercase) for i in range(len))

data = random_string(1024)
def main():
    sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sender.connect(("127.0.0.1", 8888))
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # tcp no delay 
    sender.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # congestion control algorithm
    sender.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, "scubic".encode())
    print("connection from {} to {}".format(sender.getsockname(), sender.getpeername()))
    try:
        while True:
            sender.send(data.encode())
    except KeyboardInterrupt:
        print("Caught keyboard interrupt, exiting")
        
if __name__ == "__main__":
    main()
