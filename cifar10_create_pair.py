import torch.nn as nn
import torch
import os
import numpy as np
import torch
import torchvision
import argparse
from joblib import Parallel, delayed

from utils import aux_tools


def sub_main(m, p, trial):
    data_loader, dataset = aux_tools.load_data('cifar10-simsiam', batch_size=128, \
            num_workers=0, m=m, include_dataset=True, p=p, trial=trial)
    class_num = 10



def main():
    m = 10000
    for i in range(10):
        sub_main(m, -1, i)
    # for i in range(5):
    #     sub_main(0, i)
    # for i in range(5):
    #     sub_main(1, i)
    # for i in range(5):
    #     sub_main(10, i)


if __name__ == "__main__":
    main()

