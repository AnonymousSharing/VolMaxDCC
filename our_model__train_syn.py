import torch.nn as nn
import torch
import torchvision
import argparse
import socket
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from our_model_training_utils_syn import sub_main, main


if __name__ == "__main__":
    start = time.time()
    # m = [2000, 4000, 6000, 8000]
    m = range(1000, 10001, 1000)
    # m = 8000
    # lam = 1e-5
    p = -1
    trial = 0
    list_hiddens = [3, 128, 128, 3]
    dataset_name = 'syn1-noise'

    # sub_main(lam, True, p, trial, m, list_hiddens, dataset_name, epochs=500, lr=1e-2)

    main(m, dataset_name)
    end = time.time()
    print(f'Total duration: {end-start:.2f}')

