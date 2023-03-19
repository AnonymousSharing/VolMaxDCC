import torch.nn as nn
import torch
import os
import numpy as np
import torch
import torchvision
import argparse
from joblib import Parallel, delayed

from utils import aux_tools


def sub_main(p, trial):
    data_loader, dataset = aux_tools.load_data('imagenet10-real-e600', batch_size=128, \
            num_workers=0, m=10000, include_dataset=True, p=p, trial=trial)
    class_num = 10


def main():
    for i in range(5):
        sub_main('a', i)
    for i in range(5):
        sub_main(-1, i)
    # for i in range(5):
    #     sub_main('c', i)
    # for i in range(5):
    #     sub_main(-1, i)


if __name__ == "__main__":
    main()

