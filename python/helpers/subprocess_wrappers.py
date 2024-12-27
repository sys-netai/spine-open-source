import sys
import subprocess

from . import context
from helpers.logger import LOG_LEVEL, logger


def print_cmd(cmd):
    allow_print = True if LOG_LEVEL in ["TRACE", "DEBUG"] else False
    if not allow_print:
        return
    if isinstance(cmd, list):
        cmd_to_print = " ".join(cmd).strip()
    elif isinstance(cmd, str):
        cmd_to_print = cmd.strip()
    else:
        cmd_to_print = ""

    if cmd_to_print:
        sys.stderr.write("$ %s\n" % cmd_to_print)
        # logger.debug("{}\n".format(cmd_to_print))


def call(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.call(cmd, **kwargs)


def check_call(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.check_call(cmd, **kwargs)


def check_output(cmd, **kwargs):
    print_cmd(cmd)
    # check_output returns bytes stream
    return subprocess.check_output(cmd, **kwargs).decode("utf-8")


def Popen(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.Popen(cmd, encoding="utf-8", **kwargs)
