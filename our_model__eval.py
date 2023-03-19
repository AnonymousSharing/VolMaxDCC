import os
import argparse
import torch
import torchvision
import numpy as np
from joblib import Parallel, delayed
import pickle

from our_model import OurModel
from my_dataset import StandardDataset
from evaluation import unified_metrics
from utils import aux_tools


np.set_printoptions(suppress=True, precision=4)


def direct_inference(x, y, model, device):
    model.eval()
    predictions = []
    labels_vector = []
    x = x.to(device)
    with torch.no_grad():
        c = model.forward_cluster(x)
    c = c.detach()
    predictions.extend(c.cpu().detach().numpy())
    labels_vector.extend(y.numpy())
    predictions = np.array(predictions)
    labels_vector = np.array(labels_vector)
    # print("Features shape {}".format(predictions.shape))
    return predictions, labels_vector

def inference(loader, model, device):
    model.eval()
    predictions = []
    labels_vector = []
    probs = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_single(x)
            probs.append(c.cpu().numpy())
        c = c.argmax(dim=-1)
        predictions.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(loader)}]\t Computing features...")
    predictions = np.array(predictions)
    labels_vector = np.array(labels_vector)
    probs = np.concatenate(probs)
    # print("Features shape {}".format(predictions.shape))
    return predictions, labels_vector, probs


def sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, X, y, epochs):
    device = torch.device("cuda")

    # train_dataset = StandardDataset(X[:10000], y[:10000])
    # eval_dataset = StandardDataset(X[10000:15000, :], y[10000:15000])
    # test_dataset = StandardDataset(X[15000:], y[15000:])
    train_dataset = StandardDataset(X[:10000], y[:10000])
    eval_dataset = StandardDataset(X[10000:12000, :], y[10000:12000])
    test_dataset = StandardDataset(X[12000:], y[12000:])

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        )
    data_loader_eval = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        )
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        )
    class_num=10


    train_results = []
    eval_results = []
    test_results = []

    for epoch in range(0, epochs, 5):
        # initialize model
        model = OurModel(list_hiddens, is_B_trainable)
        B_init = model._get_B().detach().cpu().numpy()

        code_name = f'our_model-ds={dataset_name}-m={m}-noise={p:s}-trial={trial:d}-L={len(list_hiddens)}'
        if is_B_trainable:
            code_name = f'{code_name}-lam={lam:e}'

        trained_model_path = os.path.join('save/%s' % (code_name), 'checkpoint_{}.tar'.format (epoch))
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['net'])
        model = model.to('cuda')

        # print('TRAIN')
        label_pred, label_true, M_pred = inference(data_loader_train, model, device)
        label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        train_results.append((nmi, ari, acc))
        # print(f'NMI: {nmi} ARI: {ari} ACC: {acc}')

        # print('EVAl')
        label_pred, label_true, M_pred = inference(data_loader_eval, model, device)
        label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        eval_results.append((nmi, ari, acc))
        # print(f'NMI: {nmi} ARI: {ari} ACC: {acc}')

        # print('TEST')
        label_pred, label_true, M_pred = inference(data_loader_test, model, device)
        label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        test_results.append((nmi, ari, acc))
        # print(f'NMI: {nmi} ARI: {ari} ACC: {acc}')

    ind = np.argmax([acc for (_, _, acc) in eval_results])
    print(f'best epoch: {ind*5}')
    # print(train_results)
    return (train_results[ind], test_results[ind], ind, eval_results[ind])


def main(m, dataset_name):
    print('m=%d' % m)
    list_lam = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    list_list_hiddens = [[512, 512, 512, 10],]
    list_p = ['-1']
    num_trials = 1
    epochs = 150
    X = np.load('datasets/imagenet-10/feature.npy')
    y = np.load('datasets/imagenet-10/label.npy')
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))

    print('#######################')
    print('NO REG ')
    print('#######################')
    is_B_trainable = False
    with Parallel(n_jobs=10) as parallel:
        for p in list_p:
            print('############ p=%s ############' % p)
            for list_hiddens in list_list_hiddens:
                all_trials = []
                results = parallel(delayed(sub_main)(0.0, is_B_trainable, p, trial, m, list_hiddens, dataset_name, X, y, epochs) for trial in range(num_trials))
                all_trials = [(train_result[2], test_result[2], train_result[1], test_result[1], train_result[0], test_result[0]) for (train_result,test_result,_,_) in results]
                all_trials = np.array(all_trials)
                means = all_trials.mean(0)
                stds = all_trials.std(0)
                print(f'L={len(list_hiddens)} (acc, nmi, ari) = ({means[0]:.4f}±{stds[0]:.4f}, {means[1]:.4f}±{stds[1]:.4f} & {means[2]:.4f}±{stds[2]:.4f}, {means[3]:.4f}±{stds[3]:.4f} & {means[4]:.4f}±{stds[4]:.4f}, {means[5]:.4f}±{stds[5]:.4f})')
                print()


    print('#######################')
    print('REG ')
    print('#######################')
    is_B_trainable = True
    with Parallel(n_jobs=6) as parallel:
        for p in list_p:
            print('############ p=%s ############' % p)
            for list_hiddens in list_list_hiddens:
                all_trials = []
                for trial in range(num_trials):
                    results = parallel(delayed(sub_main)(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, X, y, epochs) for lam in list_lam)
                    best_lam_ind = np.argmax([eval_result[2] for (_, _, _, eval_result) in results])
                    print(f'Trial {trial} has the best lam={list_lam[best_lam_ind]}')
                    train_result, test_result, _, _ = results[best_lam_ind]
                    all_trials.append((train_result[2], test_result[2], train_result[1], test_result[1], train_result[0], test_result[0]))
                all_trials = np.array(all_trials)
                means = all_trials.mean(0)
                stds = all_trials.std(0)
                print(f'L={len(list_hiddens)} (acc, nmi, ari) = ({means[0]:.4f}±{stds[0]:.4f}, {means[1]:.4f}±{stds[1]:.4f} & {means[2]:.4f}±{stds[2]:.4f}, {means[3]:.4f}±{stds[3]:.4f} & {means[4]:.4f}±{stds[4]:.4f}, {means[5]:.4f}±{stds[5]:.4f})')
                print(f'L={len(list_hiddens)} (acc, nmi, ari) = ({means[0]:.2f}, {means[1]:.2f} & {means[2]:.2f}, {means[3]:.2f} & {means[4]:.2f}, {means[5]:.2f})')
                print()


if __name__ == "__main__":
    m = 10000
    dataset_name = 'imagenet10-real'
    print(f'dataset: {dataset_name}')
    main(m, dataset_name)


