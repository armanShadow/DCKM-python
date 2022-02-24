import os

import numpy as np
import scipy.io
from tabulate import tabulate


def read_dataset(path):
    mat = scipy.io.loadmat(path)
    return mat['X_train'], mat['label'].reshape(mat['label'].shape[0])


# update centroids
def update_f(X, w, G):
    ones = np.ones((X.shape[1], 1))
    first_term = np.multiply(X.T, np.matmul(ones, w.T))
    ones_k = np.ones((G.shape[1], 1))
    second_term = np.multiply(G.T, np.matmul(ones_k, w.T))
    third_term = np.matmul(second_term, G)
    forth_term = np.matmul(first_term, G)
    return np.matmul(forth_term, np.linalg.inv(third_term))


# update cluster assignment matrix
def update_g(F, X):
    G = np.zeros((X.shape[0], F.shape[1]))
    for i, sample in enumerate(X):
        distances = []
        for centroid in F.T:
            distances += [np.linalg.norm(sample - centroid)]
        minimum = min(distances)
        index = distances.index(minimum)
        gi = np.zeros((1, F.shape[1]))
        gi[0][index] = 1
        G[i] = gi
    return G

def DCKM():
    w = np.array([[1], [2], [1.5]])
    X = np.array([[1, 2, 3], [2, 1, 5], [5, 2, 1]])
    G = np.array([[1, 0], [0, 1], [0, 1]])
    F = update_f(X, w, G)
    new_g = update_g(F, X)
    return 0, 0


def main():
    datasets = os.listdir('data/')
    metrics = []
    for dataset in datasets:
        X_train, y_train = read_dataset(f'data\{dataset}')
        nmi, ari = DCKM(X_train, y_train)
        metrics += [[dataset, nmi, ari]]

    table = tabulate(metrics, headers=['Dataset', 'NMI', 'ARI'])
    print(table)


if __name__ == '__main__':
    DCKM()
