from logging import log
import os
from os import environ, path
from subprocess import check_call
import sys

import arg_parser
import context
from context import src_dir


# from model.current.total_config import exp_config



def main():
    args = arg_parser.receiver_first()

    cc_dir = "astraea"
    cc_repo = path.abspath(path.join(context.src_dir, cc_dir))
    build_src = path.join(cc_repo, "build")
    bin_dir = path.join(cc_repo, "build", "bin")
    recv_src = path.join(bin_dir, "server")
    active_send_src = path.join(bin_dir, "client")
    eval_sender_src = path.join(bin_dir, "client_spine")

    # variables for inference
    pyhelper = path.join(context.src_dir, "eval", "spine_infer.py")
    model_path = path.join(context.src_dir, "model", "current", "ckpt_best.pth.tar")
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    os.environ["LOG_LEVEL"] = log_level

    # allow to dump cwnd
    # os.environ["PLOT_CWND"] = "1"

    if args.option == "setup":
        check_call(["mkdir", "-p", "build"], cwd=cc_repo)
        check_call(["cmake", ".."], cwd=build_src)
        check_call(["make"], cwd=build_src)
        return

    if args.option == "receiver":
        cmd = [recv_src, "--port={}".format(args.port)]
        check_call(cmd)
        return

    if args.option == "sender":
        if not args.eval:
            send_src = active_send_src
            cmd = [
                send_src,
                "--ip={}".format(args.ip),
                "--port={}".format(args.port),
                "--cong={}".format(args.cong),
                "--ipc={}".format(args.ipc),
                "--interval={}".format(args.interval),
            ]
            # for debug
            # cmd += ["|", "tee", "/tmp/astraea-flow-{}.log".format(args.id)]
        else:
            send_src = eval_sender_src
            cmd = [
                send_src,
                "--ip={}".format(args.ip),
                "--port={}".format(args.port),
                "--cong={}".format(args.cong),
                "--pyhelper={}".format(pyhelper),
                "--model={}".format(model_path),
                "--interval={}".format(args.interval),
            ]
        if args.id != None:
            cmd += ["--id={}".format(args.id)]

        # if args.perf_log != None:
        #     c:whmd += ["--perf-log={}".format(args.perf_log)]

        sys.stderr.write("{}\n".format(cmd))
        check_call(cmd)
        return


if __name__ == "__main__":
    main()
