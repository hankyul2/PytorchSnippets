# https://pytorch.org/docs/stable/notes/randomness.html
# 1. seed_everything
# 2. deterministic algorithm
# 3. seed worker_init (you should also set generator)

import torch
import random
import numpy as np


def seed_everything(seed, deterministic):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


if __name__ == '__main__':
    seed_everything(5, deterministic=True)
