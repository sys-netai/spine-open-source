from functools import partial
import os
import sys
import argparse
import signal

import context
from helpers.logger import logger
from agent.master import Master


def main(args):
    master = Master(args.ipc, args.threads, dump_state=True)
    master.run()

    def signal_handler(signum, frame):
        logger.fatal("RL master caught signal: {}, exiting...".format(signum))
        master.dump_data("data.json")
        sys.exit("Master forced to quit")

    # register signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # signal.signal(signal.SIGKILL, signal_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--ipc", help="IPC socket between env", type=str, required=True
    )
    parser.add_argument(
        "-t", "--threads", help="number of threads", type=int, default=os.cpu_count()
    )
    args = parser.parse_args()

    main(args)
