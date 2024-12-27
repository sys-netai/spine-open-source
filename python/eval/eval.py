import os
import json
import time

import context

from env.env import MultiFlowEnvCore
from helpers.logger import logger


def main(config):
    role = "eval-actor"
    task_id = 1
    env_id = "{}-{}".format(role, task_id)
    env = MultiFlowEnvCore(
        inside_alg=True,
        sample_action=None,
        active_sender=True,
        allow_halt_flow=True,
        evaluate=True,
        env_id=env_id,
        eval_mode=True,
    )
    conf = None
    with open(config, "r") as f:
        conf = json.load(f)

    for episode in range(8):
        conf["flows"] = episode + 1
        env.reset()
        env.make_world(conf)
        env.run()

    # before exit, we need to call
    env.exit()


if __name__ == "__main__":
    config = "eval.json"
    main(config)
