import os
import sys
from enum import Enum


class MessageType(Enum):
    INIT = 0  # env initialization
    START = 1  # episode start
    END = 2  # episode end
    ALIVE = 3  # alive status
    OBSERVE = 4  # observe the world
    TERMINATE = 5  # terminate the env
    MEASURE = 6


