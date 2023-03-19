import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
from scipy.io import savemat


def plot_simplex(M):
    """
    M: N x K
    vertices: K x 2
    """
    vertices = np.array([[-0.7, 0], [0.7, 0], [0, 1]])
    new_coor = np.matmul(M.T,vertices)
    plt.figure()
    colors = [(1.0, a, b)  for a, b in zip(M[0, :], M[1, :])]
    plt.scatter(new_coor[:, 0], new_coor[:, 1], marker='x', color=colors)
    plt.title('K = %d' % M.shape[0])
    plt.savefig('./datasets/synthetic/membership.pdf', bbox_inches='tight')

def f_inv(M):
    """
    M: 3 x N
    """
    X = np.zeros(M.shape)
    X[0, :] = 2*M[0, :]
    X[1, :] = 3*M[1, :] +1
    X[2, :] = M[0, :] * M[1, :] - 2
    # K, N = M.shape
    # X = np.random.randn(K, K)@np.log(M) + np.random.randn(K, 1)
    return X

def generate_data(m):
    K = 3
    N = 1000
    rate = 2

    M = np.zeros((K, int(N*rate)))
    M[np.random.randint(0, 3, int(N*rate)), np.arange(int(N*rate))] = 1
    M = M + 0.1*np.random.randn(K, int(N*rate))
    M[M<0] = 0
    M = M/M.sum(0)
    plot_simplex(M)

    rand_i = np.random.randint(0, N, size=m)
    rand_j = np.random.randint(0, N, size=m)
    m_i = M[:, rand_i]
    m_j = M[:, rand_j]
    # B = np.eye(3)
    B = np.diag([0.95, 0.95, 0.95]) + 0.2*np.random.randn(3, 3)
    B[B>1] = 1.
    B[B<0] = 0.
    P = np.matmul(np.matmul(M.T, B), M)
    y = (np.random.rand(m) <= P[rand_i, rand_j]).astype(np.float32)
    X = f_inv(M).astype(np.float32)
    return rand_i, rand_j, y, X, M, B, N


if __name__ == "__main__":

    list_m = list(range(1000, 10001, 1000)) + [12000, 15000, 20000]
    for m in list_m:
        for trial in range(10):
            print(trial)
            ind_i, ind_j, pair_labels, X, M, B, N = generate_data(m)
            data = dict()
            data['i'] = ind_i
            data['j'] = ind_j
            data['pair_labels'] = pair_labels
            data['X'] = X
            data['M'] = M
            data['B'] = B
            data['N'] = N
            with open('datasets/synthetic/data_noise_m_%d_trial_%d.pkl' % (m, trial),  'wb') as file_handler:
                pickle.dump(data, file_handler)

