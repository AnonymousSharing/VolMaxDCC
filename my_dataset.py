from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class StandardDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, ind):
        return self.X[ind, :], self.y[ind]


class MyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sub_dataset, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = sub_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item, label = self.data[idx]
        item = self.transform(item)
        return item, label



class CustomDataset(Dataset):
    def __init__(self, dataset, transform):
        self.trans = transform
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return (self.trans(item[0]), item[1])

    def __len__(self):
        return len(self.dataset)


class AugmentedRandomPairDataset(Dataset):
    def __init__(self, X, ind_pairs, label_pairs, transforms):
        self.trans = transforms
        self.X = X
        self.ind_pairs = ind_pairs
        self.label_pairs = label_pairs
        print('AugmentedRandomPairDataset: Number of pair: %d' % len(label_pairs))

    def __getitem__(self, index):
        i1, i2 = self.ind_pairs[index]
        x1 = self.X[i1]
        x1 = self.trans(x1)
        x2 = self.X[i2]
        x2 = self.trans(x2)
        y = self.label_pairs[index]
        return (x1, x2), y

    def __len__(self):
        return len(self.ind_pairs)

class AugmentedRandomPairDataset2(Dataset):
    def __init__(self, pair_dataset, transforms):
        self.trans = transforms
        self.pair_dataset = pair_dataset
        print('AugmentedRandomPairDataset: Number of pair: %d' % len(pair_dataset))

    def __getitem__(self, index):
        (x1, x2), y = self.pair_dataset[index]
        x1 = self.trans(x1)
        x2 = self.trans(x2)
        return (x1, x2), y

    def __len__(self):
        return len(self.pair_dataset)

class SingleIndexDataset(Dataset):
    def __init__(self, sub_dataset):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = sub_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item, label = self.data[idx]
        return item, idx
