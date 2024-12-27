import os
from os import path
import sys
src_dir = path.abspath(path.join(path.dirname(__file__), os.pardir, "src"))
print("add src dir: %s", src_dir)
log_dir = path.abspath(path.join(src_dir, os.pardir, "log"))
sys.path.append(src_dir)