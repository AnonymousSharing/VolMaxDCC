import random
import os
import itertools

import numpy as np
import torch
import torchvision
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

from stl10_oracle__train import STL10Oracle
from cifar10_oracle__train import CIFAR10Oracle
from imagenet10_oracle__train import ImageNet10Oracle


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_accuracy(pred, y):
    pred = pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    return np.mean(pred==y)

def clust_acc(pred, y, no_comm):
    classes = pred
    cost = np.zeros((no_comm, no_comm))
    for i in range(no_comm):
        for j in range(no_comm):
            cost[i, j] = -np.sum((classes == i) & (y == j))

    row_ind, col_ind = linear_sum_assignment(cost)
    classes = [col_ind[item] for item in classes]
    return np.mean(y == classes)

def load_data(name, batch_size=32, num_workers=8, p=0.001, include_dataset=False, e=0.01, m=-1,trial=-1):
    if name=='cifar10-real':
        data_pairs = load_cifar10_real(m, p, trial)
    elif name=='cifar10-simsiam':
        data_pairs = load_cifar10_simsiam(m, p, trial)
    elif name=='imagenet10-real':
        data_pairs = load_imagenet10_real(m, p, trial)
    elif name=='imagenet10-real-e1000':
        data_pairs = load_imagenet10_real_e1000(m, p, trial)
    elif name=='imagenet10-real-e600':
        data_pairs = load_imagenet10_real_e600(m, p, trial)
    elif name=='imagenet10-real-e100':
        data_pairs = load_imagenet10_real_e100(m, p, trial)
    elif name=='imagenet10-real-e50':
        data_pairs = load_imagenet10_real_e50(m, p, trial)
    elif name=='stl10-real':
        data_pairs = load_stl10_real(m, p, trial)
    elif name=='stl10-byol':
        data_pairs = load_stl10_byol(m, p, trial)
    elif name=='syn1':
        data_pairs = load_syn1(m, p, trial)
    elif name=='syn1-noise':
        data_pairs = load_syn1_noise(m, p, trial)
    else:
        raise Exception('typo')

    data_loader = torch.utils.data.DataLoader(
        data_pairs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    if include_dataset:
        return data_loader, data_pairs
    else:
        return data_loader

class PairDataset(Dataset):
    def __init__(self, ind_pairs, label_pairs, X, others=[]):
        self.ind_pairs = ind_pairs
        self.label_pairs = label_pairs
        self.X = X
        self.others = others

    def __len__(self):
        return len(self.ind_pairs)

    def __getitem__(self, idx):
        i1, i2 = self.ind_pairs[idx]
        return (self.X[i1], self.X[i2]), self.label_pairs[idx]

    def getitem_extra(self, idx):
        i1, i2 = self.ind_pairs[idx]
        return (self.X[i1], self.X[i2]), self.label_pairs[idx], (i1, i2, self.others[idx])

def freeze_it(model):
    for param in model.parameters():
        param.requires_grad = False

def load_stl10_real(m, p, trial):
    pair_path = 'datasets/stl10/pair_%d_real_%d_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        true_label_pairs = data_dict['true_label_pairs']
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()

        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
    else:
        print('Loading raw dataset')
        X = np.load('datasets/stl10/feature.npy')[:10000]
        X = torch.from_numpy(X.astype(np.float32))
        true_y = np.load('datasets/stl10/label.npy')[:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        i1 = np.random.randint(0, N, size=m).tolist()
        random.shuffle(i1)
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y=[]
        if p!=-1:
            model = STL10Oracle(512, 10)
            trained_model_path = os.path.join('save/stl10-oracle/', "checkpoint_{:d}.tar".format(p))
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()


        # ind_pairs, label_pairs = make_pairs_random_2(y, m)
        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')
    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset

def load_stl10_byol(m, p, trial):
    pair_path = 'datasets/stl10/pair_%d_byol_%d_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        with open('./../byol-pytorch/datasets/stl10/data.pkl', 'rb') as file_input:
            data_dict = pickle.load(file_input)
            X = torch.from_numpy(data_dict['X'].astype(np.float32))[:10000]
            true_y = data_dict['Y'][:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        i1 = np.random.randint(0, N, size=m).tolist()
        random.shuffle(i1)
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y = []
        if p!=-1:
            __import__('pdb').set_trace()
            model = CIFAR10Oracle(512, 10)
            trained_model_path = os.path.join('save/cifar10-oracle/', "checkpoint_{}.tar".format(p))
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()


        # ind_pairs, label_pairs = make_pairs_random_2(y, m)
        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')
    print(f'Positive proportion: {label_pairs.mean():.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset

def load_cifar10_real(m, p, trial):
    pair_path = 'datasets/cifar10/pair_%d_real_%d_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        X = np.load('datasets/cifar10/feature.npy')[:10000]
        X = torch.from_numpy(X.astype(np.float32))
        true_y = np.load('datasets/cifar10/label.npy')[:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        if m>N:
            i1 = list(range(N)) + np.random.randint(0, N, size=(m-N)).tolist()
        else:
            i1 = np.random.randint(0, N, size=m).tolist()
        random.shuffle(i1)
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0


        y = []
        if p!=-1:
            model = CIFAR10Oracle(512, 10)
            trained_model_path = os.path.join('save/cifar10-oracle/', "checkpoint_{}.tar".format(p))
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()


        # ind_pairs, label_pairs = make_pairs_random_2(y, m)
        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset

def load_cifar10_simsiam(m, p, trial):
    pair_path = 'datasets/cifar10/pair_%d_simsiam_%d_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        with open('./../simsiam/datasets/cifar10/data.pkl', 'rb') as file_input:
            data_dict = pickle.load(file_input)
            X = torch.from_numpy(data_dict['X'].astype(np.float32))[:10000]
            true_y = data_dict['Y'][:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        i1 = np.random.randint(0, N, size=m).tolist()
        random.shuffle(i1)
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y = []
        if p!=-1:
            __import__('pdb').set_trace()
            model = CIFAR10Oracle(512, 10)
            trained_model_path = os.path.join('save/cifar10-oracle/', "checkpoint_{}.tar".format(p))
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()


        # ind_pairs, label_pairs = make_pairs_random_2(y, m)
        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')
    print(f'Positive proportion: {label_pairs.mean():.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset

def load_imagenet10_real(m, p, trial):
    pair_path = 'datasets/imagenet-10/pair_%d_real_%s_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        X = np.load('datasets/imagenet-10/feature.npy')[:10000]
        X = torch.from_numpy(X.astype(np.float32))
        true_y = np.load('datasets/imagenet-10/label.npy')[:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        if m>N:
            i1 = list(range(N)) + np.random.randint(0, N, size=(m-N)).tolist()
        else:
            i1 = np.random.randint(0, N, size=m).tolist()
        random.shuffle(i1)
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y= []
        if str(p)!='-1':
            model = ImageNet10Oracle(512, 10)
            trained_model_path = os.path.join('save/imagenet10-oracle/', "checkpoint_{:s}.tar".format(p)) #[2, 3, 9]
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()

        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset

def load_imagenet10_real_e1000(m, p, trial):
    pair_path = 'datasets/imagenet-10/pair_e1000_%d_%s_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        X = np.load('datasets/imagenet-10/feature_1000.npy')[:10000]
        X = torch.from_numpy(X.astype(np.float32))
        true_y = np.load('datasets/imagenet-10/label_1000.npy')[:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        i1 = np.random.randint(0, N, size=m).tolist()
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y= []
        if p!=-1:
            model = ImageNet10Oracle(512, 10)
            trained_model_path = os.path.join('save/imagenet10-oracle-e1000/', "checkpoint_%s.tar" % (p)) #[a, b, c]
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()

        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset
def load_imagenet10_real_e600(m, p, trial):
    pair_path = 'datasets/imagenet-10/pair_e600_%d_%s_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        X = np.load('datasets/imagenet-10/feature_600.npy')[:10000]
        X = torch.from_numpy(X.astype(np.float32))
        true_y = np.load('datasets/imagenet-10/label_600.npy')[:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        i1 = np.random.randint(0, N, size=m).tolist()
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y= []
        if p!=-1:
            model = ImageNet10Oracle(512, 10)
            trained_model_path = os.path.join('save/imagenet10-oracle-e600/', "checkpoint_%s.tar" % (p)) #[a, b, c]
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()

        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset
def load_imagenet10_real_e100(m, p, trial):
    pair_path = 'datasets/imagenet-10/pair_e100_%d_%s_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        X = np.load('datasets/imagenet-10/feature_100.npy')[:10000]
        X = torch.from_numpy(X.astype(np.float32))
        true_y = np.load('datasets/imagenet-10/label_100.npy')[:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        i1 = np.random.randint(0, N, size=m).tolist()
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y= []
        if p!=-1:
            model = ImageNet10Oracle(512, 10)
            trained_model_path = os.path.join('save/imagenet10-oracle-100/', "checkpoint_%s.tar" % (p)) #[a, b, c]
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()

        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset
def load_imagenet10_real_e50(m, p, trial):
    pair_path = 'datasets/imagenet-10/pair_e50_%d_%s_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        X = np.load('datasets/imagenet-10/feature_50.npy')[:10000]
        X = torch.from_numpy(X.astype(np.float32))
        true_y = np.load('datasets/imagenet-10/label_50.npy')[:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        i1 = np.random.randint(0, N, size=m).tolist()
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y= []
        if p!=-1:
            model = ImageNet10Oracle(512, 10)
            trained_model_path = os.path.join('save/imagenet10-oracle-e50/', "checkpoint_%s.tar" % (p)) #[a, b, c]
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()

        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset

def match_it(M1, M2, criterion):
    """
    :M1: K x N
    :M2: K x N
    :returns: M1 ~= PI*M2
    """
    if criterion == 'RE':
        M1 = M1/(np.sqrt((M1.power(2)).sum(1)))
        M2 = M2/(np.sqrt((M2**2).sum(1, keepdims=True))+1e-9*np.ones(M2.shape[1]))

        K = M1.shape[0]
        cost = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                tmp = M1[i, :] - M2[j, :]
                cost[i, j] = (np.power(tmp, 2)).sum()
        _, best_ind = linear_sum_assignment(cost)
        new_M2 = M2[best_ind, :]
        PI = np.zeros((K, K))
        for i in range(K):
            PI[i, best_ind[i]] = 1
    else:
        raise Exception('Typing')

    return new_M2, best_ind, PI

def load_syn1(m, p, trial):
    print('Loading raw dataset')
    with open('datasets/synthetic/data_m_%d_trial_%d.pkl' % (m, trial), 'rb') as input_file:
        data = pickle.load(input_file)
    X = torch.from_numpy(data['X'].T)
    M = torch.from_numpy(data['M'])
    assert ((data['i']>=data['N']).sum()) == 0
    assert ((data['j']>=data['N']).sum()) == 0
    ind_pairs = list(zip(data['i'], data['j']))
    true_label_pairs = data['pair_labels']
    label_pairs = torch.from_numpy(true_label_pairs.astype(np.float32))

    print('#Nodes: %d \t #Edges: %d \n' % (data['N'], len(label_pairs)))
    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [])
    return pair_dataset

def load_syn1_noise(m, p, trial):
    print('Loading raw dataset')
    with open('datasets/synthetic/data_noise_m_%d_trial_%d.pkl' % (m, trial), 'rb') as input_file:
        data = pickle.load(input_file)
    X = torch.from_numpy(data['X'].T)
    M = torch.from_numpy(data['M'])
    assert ((data['i']>=data['N']).sum()) == 0
    assert ((data['j']>=data['N']).sum()) == 0
    ind_pairs = list(zip(data['i'], data['j']))
    true_label_pairs = data['pair_labels']
    label_pairs = torch.from_numpy(true_label_pairs.astype(np.float32))

    print('#Nodes: %d \t #Edges: %d \n' % (data['N'], len(label_pairs)))
    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [])
    return pair_dataset

