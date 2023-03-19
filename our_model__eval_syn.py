import os
import argparse
import torch
import torchvision
import numpy as np
from joblib import Parallel, delayed
import pickle
import matplotlib.pyplot as plt

from our_model import OurModel
from my_dataset import StandardDataset
from evaluation import unified_metrics
from utils import aux_tools


np.set_printoptions(suppress=True, precision=4)


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


def sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs):
    device = torch.device("cuda")

    with open('datasets/synthetic/data_noise_m_%d_trial_%d.pkl' % (m, trial), 'rb') as input_file:
        data = pickle.load(input_file)
    X = torch.from_numpy(data['X'].T)
    M = torch.from_numpy(data['M'])
    N = data['N']
    M_true_train = M[:, :N]
    M_true_test = M[:, N:]
    ind_pairs = list(zip(data['i'], data['j']))
    true_label_pairs = data['pair_labels']
    label_pairs = torch.from_numpy(true_label_pairs.astype(np.float32))

    train_dataset = StandardDataset(X[:N], M[:, :N].T)
    test_dataset = StandardDataset(X[N:], M[:, N:].T)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
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
    class_num=3


    train_results = []
    test_results = []

    for epoch in range(0, epochs+1, 5):
        # initialize model
        model = OurModel(list_hiddens, is_B_trainable)
        B_init = model._get_B().detach().cpu().numpy()

        code_name = f'our_model-ds={dataset_name}-m={m}-noise={p:d}-trial={trial:d}-L={len(list_hiddens)}'
        if is_B_trainable:
            code_name = f'{code_name}-lam={lam:e}'

        trained_model_path = os.path.join('save/%s' % (code_name), 'checkpoint_{}.tar'.format (epoch))
        print(f'Load checkpoint at {trained_model_path}')
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['net'])
        model = model.to('cuda')

        _, _, M_pred = inference(data_loader_train, model, device)
        M_pred = M_pred.T
        mse_train = unified_metrics.MSE(M_true_train, M_pred)
        mse_train = mse_train.item()
        train_results.append(mse_train)
        print(f'Train({len(train_dataset)})- Epoch: {epoch} - Rel error: {mse_train:e}')

        _, _, M_pred = inference(data_loader_test, model, device)
        M_pred = M_pred.T
        mse_test = unified_metrics.MSE(M_true_test, M_pred)
        mse_test = mse_test.item()
        test_results.append(mse_test)
        print(f'Test({len(test_dataset)}) - Epoch: {epoch} - Rel error: {mse_test:e}')

    best_ind = np.argmin(test_results)
    print(f'Best epoch {best_ind} with test={test_results[best_ind]:e}')
    return train_results[best_ind], test_results[best_ind]

        # # print('EVAl')
        # label_pred, label_true, M_pred = inference(data_loader_eval, model, device)
        # label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        # nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        # eval_results.append((nmi, ari, acc))
        # # print(f'NMI: {nmi} ARI: {ari} ACC: {acc}')
        #
        # # print('TEST')
        # label_pred, label_true, M_pred = inference(data_loader_test, model, device)
        # label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        # nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        # test_results.append((nmi, ari, acc))
        # # print(f'NMI: {nmi} ARI: {ari} ACC: {acc}')

    # ind = np.argmax([acc for (_, _, acc) in eval_results])
    # print(f'best epoch: {ind*5}')
    # print(train_results)
    # return (train_results[ind], test_results[ind], ind, eval_results[ind])


def main(list_m, dataset_name, lam):
    list_hiddens = [3, 128, 128, 3]
    list_p = -1
    num_trials = 10
    epochs = 500
    trial = 0

    is_B_trainable = True
    # is_B_trainable = False

    all_results = []
    for m in list_m:
        with Parallel(n_jobs=6) as parallel:
            results = parallel(delayed(sub_main)(lam, is_B_trainable, -1, trial, m, list_hiddens, dataset_name, epochs) for trial in range(num_trials))
        results = np.array(results)
        all_results.append((m, results))

    return all_results



if __name__ == "__main__":
    m = list(range(1000, 10001, 1000)) + [12000, 15000, 20000]
    m = list(range(1000, 10001, 1000))
    dataset_name = 'syn1-noise'
    lam = 1e-4
    print(f'dataset: {dataset_name}')
    results = main(m, dataset_name, lam)
    with open('syn_noise_result_%e.pkl' % lam, 'wb') as output_file:
        pickle.dump(results, output_file)
    # with open('syn_noise_result_withoutB.pkl', 'wb') as output_file:
    #     pickle.dump(results, output_file)




    # lam = 0.0
    # is_B_trainable = True
    # p = -1
    # trial = 1
    # m = 12000
    # list_hiddens = [3, 128, 128, 3]
    # dataset_name = 'syn1-noise'
    # epochs = 500
    # sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs)

