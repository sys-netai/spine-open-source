import os
import sys
import multiprocessing
from multiprocessing import Pool

from . import context

def sample_action(alg, state, tun_id):
    action = alg(state, tun_id)
    # may need train
    need_train = True
    if need_train:
        # this will return immediately 
        Pool.apply_async(alg.train())
    return action

def dummy_sample_action(alg, state, tun_id):
    return 400
