import os
import sys
from os import path

env_dir = path.abspath(path.join(path.dirname(__file__)))
src_dir = path.abspath(path.join(env_dir, os.pardir, os.pardir))
sys.path.append(src_dir)