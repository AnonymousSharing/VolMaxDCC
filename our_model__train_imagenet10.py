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
    dataset_name = 'imagenet10-real'

    main(m, dataset_name)
    end = time.time()
    print(f'Total duration: {end-start:.2f}')

