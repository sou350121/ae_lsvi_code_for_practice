"""
Launch a bunch of jobs.
"""
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import subprocess
import time
from typing import Any, Dict

import numpy as np


log_path = 'rl_train_many.txt'
max_running = 8
max_jobs_per_gpu = 2
gpu_counts = [0 for _ in range(4)]
running = []
name_field = 'env_name'

# Args
arg_dict = OrderedDict({
    'env': [
        "denseuniformcartpoleswingup-v0",
        "densemountaincar-dt10-v0",
        "uniformweirdgain-v0",
        "uniformbetarotation-v0",
        "uniformbetatracking-v0",
        "newbetarotation-v0",
        "betatracking-v0",
    ],
    'seed': [t for t in range(5)],
})


@dataclass
class Job:
    proc: subprocess.Popen
    gpu: int
    args: Dict[str, Any]


def prune_completed_job():
    for jidx, job in enumerate(running):
        if job.proc.poll() is not None:
            gpu_counts[job.gpu] -= 1
            with open(log_path, 'a') as f:
                f.write(f'{datetime.now()}\t Finished \t {job.args}\n')
            running.pop(jidx)
            return True
    return False


def add_job(args):
    if len(running) >= max_running:
        while not prune_completed_job():
            time.sleep(30)
    with open(log_path, 'a') as f:
        f.write(f'{datetime.now()}\t Starting \t {args}\n')
    # Find open gpu device.
    gpu = 0
    while gpu < len(gpu_counts) - 1 and gpu_counts[gpu] >= max_jobs_per_gpu:
        gpu += 1
    gpu_counts[gpu] += 1
    cmd = 'python scripts/train_dqn.py '
    cmd += ' '.join([f'--{k} {v}' for k, v in args.items()])
    cmd += f' --cuda_device {gpu}'
    proc = subprocess.Popen(cmd, shell=True)
    running.append(Job(proc, gpu, args))


with open(log_path, 'w') as f:
    f.write('Timestamp \t Status \t Args\n')
arg_keys = list(arg_dict.keys())
num_each_args = np.array([len(arg_dict[k]) for k in arg_keys])
arg_idxs = np.array([0 for _ in range(len(arg_dict))])
while True:
    add_job({k: arg_dict[k][arg_idxs[kidx]] for kidx, k, in enumerate(arg_keys)})
    arg_idxs[0] += 1
    for ii in range(len(arg_idxs) - 1):
        if arg_idxs[ii] >= num_each_args[ii]:
            arg_idxs[ii] = 0
            arg_idxs[ii + 1] += 1
    if np.any(arg_idxs >= num_each_args):
        break
while len(running) > 0:
    prune_completed_job()
    time.sleep(30)
