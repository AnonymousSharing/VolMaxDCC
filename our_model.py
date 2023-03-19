import itertools
from collections import OrderedDict

import torch.nn as nn
import torch
import torchvision
import socket
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed



class OurModel(nn.Module):
    def __init__(self, list_hiddens, is_B_trainable=True):
        super(OurModel, self).__init__()
        self.normalized_loss = nn.BCELoss(reduction='sum')
        list_layers = [[('linear'+str(i), \
                            nn.Linear(list_hiddens[i], list_hiddens[i+1])), \
                        ('relu'+str(i), nn.ReLU())] \
                        for i in range(len(list_hiddens)-1)]
        list_layers = list(itertools.chain(*list_layers))
        list_layers[-1] = ('final_softmax', nn.Softmax(dim=1))
        self.cluster_projector = nn.Sequential(OrderedDict(list_layers))

        class_num = list_hiddens[-1]
        self.is_B_trainable = is_B_trainable

        if is_B_trainable:
            B_init = 1*torch.eye(class_num)
            B_init[B_init==0] = -1
            self._B = nn.Parameter(B_init, requires_grad=True)
            self._sigmoid = nn.Sigmoid()
        else:
            self._B = torch.eye(class_num).to('cuda')
            self._sigmoid = lambda x: x
        self.I = torch.eye(class_num).to('cuda')

    def _get_B(self):
        return self._sigmoid(self._B)

    def forward_single(self, x):
        c = self.cluster_projector(x)
        return c

    def forward(self, x_i, x_j):
        c_i = self.forward_single(x_i)
        c_j = self.forward_single(x_j)
        return c_i, c_j

    def compute_loss_1(self, B, c_i, c_j, y):
        probs = self.link_prediction(B, c_i, c_j)
        probs[probs>=1] = 1 - 1e-6
        probs[probs<=0] = 1e-6
        loss = self.normalized_loss(probs, y)
        N = c_i.shape[0]
        return loss/N

    def compute_loss_2(self, F):
        return torch.log(torch.det(torch.matmul(F.T, F) + 1e-5 * self.I))

    def compute_loss(self, B, c_i, c_j, y, lam, X):
        loss1 = self.compute_loss_1(B, c_i, c_j, y)
        if self.is_B_trainable:
            F = self.forward_single(X)
            loss2 = self.compute_loss_2(F)
            return loss1 - lam*loss2, loss1, loss2
        else:
            return loss1, loss1, loss1

    def link_prediction(self, B, c_i, c_j):
        return torch.mul(torch.matmul(c_i, B), c_j).sum(-1)

