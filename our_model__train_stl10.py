import torch.nn as nn
import torch
import torchvision
import argparse
import socket
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from our_model_training_utils import sub_main, main


if __name__ == "__main__":
    start = time.time()
    m = 10000
    lam = 0.0
    p = -1
    trial = 0
    list_hiddens = [512, 512, 512, 10]
    dataset_name = 'stl10-real'

    # sub_main(lam, False, p, trial, m, list_hiddens, dataset_name, epochs=100, lr=1e-4)

    main(m, dataset_name)
    end = time.time()
    print(f'Total duration: {end-start:.2f}')

