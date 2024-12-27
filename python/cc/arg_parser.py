import argparse
import sys


def parse_wrapper_args(run_first):
    if run_first != "receiver" and run_first != "sender":
        sys.exit('Specify "receiver" or "sender" to run first')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="option")

    subparsers.add_parser(
        "run_first", help="print which side (sender or receiver) runs first"
    )

    receiver_parser = subparsers.add_parser("receiver", help="run receiver")
    sender_parser = subparsers.add_parser("sender", help="run sender")

    if run_first == "receiver":
        receiver_parser.add_argument("port", help="port to listen on")
        sender_parser.add_argument("ip", metavar="IP", help="IP address of receiver")
        sender_parser.add_argument("port", help="port of receiver")
        # for astraea
        sender_parser.add_argument("--ipc", help="path of IPC file", default="")
        sender_parser.add_argument(
            "--cong",
            help="underlying congestion control algorithms used by Astraea",
            default="astraea",
        )
        sender_parser.add_argument(
            "--interval",
            help="control interval in milliseconds, default 10ms",
            default=10,
        )
        sender_parser.add_argument(
            "--active",
            help="Active sender or not",
            default=1,
        )
        sender_parser.add_argument(
            "--id",
            help="Current flow id, will be used in identify flows in IPC with RL master.",
            default=None,
        )
        sender_parser.add_argument(
            "-e", "--eval", action="store_true", help="Evaluation model of Astraea"
        )
        sender_parser.add_argument(
            "-l",
            "--perf-log",
            type=str,
            default=None,
            help="File name for performance log",
        )
    else:
        sender_parser.add_argument("port", help="port to listen on")
        receiver_parser.add_argument("ip", metavar="IP", help="IP address of sender")
        receiver_parser.add_argument("port", help="port of sender")

    args = parser.parse_args()

    if args.option == "run_first":
        print(run_first)

    return args


def receiver_first():
    return parse_wrapper_args("receiver")


def sender_first():
    return parse_wrapper_args("sender")
