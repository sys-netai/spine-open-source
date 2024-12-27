import os
from os import path
import sys

eval_dir = path.abspath(path.join(path.dirname(__file__)))
src_dir = path.abspath(path.join(eval_dir, os.pardir))
sys.path.append(src_dir)
