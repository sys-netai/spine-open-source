import os
from os import path
import sys
agent_dir = path.abspath(path.join(path.dirname(__file__)))
src_dir = path.abspath(path.join(agent_dir, os.pardir))
# env_dir = path.abspath(path.join(src_dir, os.pardir, "evaluate"))
sys.path.append(src_dir)