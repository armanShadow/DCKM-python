import os

import numpy as np
import scipy.io
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tabulate import tabulate

learning_rate = 0.01
lambda1 = 1
lambda2 = 1
lambda3 = 1
e = 0.01


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


def feature_decorrelation(X, j):
    uncorrelated = np.copy(X.T)
    uncorrelated[j] = np.zeros((1, X.shape[0]))
    return uncorrelated


def jb(X, w, j):
    first_term = np.matmul(feature_decorrelation(X, j), np.multiply(np.multiply(w, w), X[:, j].reshape(-1, 1)))
    second_term = np.matmul(np.multiply(w, w).T, X[:, j].reshape(-1, 1))
    third_term = np.matmul(feature_decorrelation(X, j), np.multiply(np.multiply(w, w), 1 - X[:, j].reshape(-1, 1)))
    forth_term = np.matmul(np.multiply(w, w).T, 1 - X[:, j].reshape(-1, 1))
    return (first_term / second_term) - (third_term / forth_term)


def partial_gradient_jbw(X, w, j):
    ones = np.ones((X.shape[1], 1))
    first_term_1 = np.multiply(feature_decorrelation(X, j), np.matmul(X[:, j].reshape(-1, 1), ones.T).T)
    first_term_2 = np.matmul(np.multiply(w, w).T, X[:, j].reshape(-1, 1))
    first_term = first_term_1 * first_term_2
    second_term = np.matmul(np.multiply(w, w).T, X[:, j].reshape(-1, 1)) ** 2

    third_term = np.multiply(np.multiply(w, w), X[:, j].reshape(-1, 1))
    third_term = np.matmul(feature_decorrelation(X, j), third_term)
    third_term = np.matmul(third_term, (X[:, j].reshape(-1, 1)).T)

    forth_term_1 = np.matmul(np.multiply(w, w).T, 1 - X[:, j].reshape(-1, 1))
    forth_term_2 = np.matmul(1 - X[:, j].reshape(-1, 1), ones.T).T
    forth_term = np.multiply(feature_decorrelation(X, j), forth_term_2) * forth_term_1
    fifth_term = np.matmul(np.multiply(w, w).T, 1 - X[:, j].reshape(-1, 1)) ** 2

    sixth_term_1 = np.multiply(np.multiply(w, w), 1 - X[:, j].reshape(-1, 1))
    sixth_term = np.matmul(feature_decorrelation(X, j), np.matmul(sixth_term_1, (1 - X[:, j].reshape(-1, 1)).T))

    return (first_term / second_term) - (third_term / second_term) - (forth_term / fifth_term) + (
            sixth_term / fifth_term)


def partial_gradient_jw(X, F, G, w):
    ones_d = np.ones((X.shape[1], 1))
    first_term_1 = np.multiply(X.T - np.matmul(F, G.T), X.T - np.matmul(F, G.T))
    first_term = np.multiply(np.matmul(ones_d.T, first_term_1).T, w)

    sum_1 = np.zeros((X.shape[0], 1))
    for j in range(X.shape[1]):
        second_term_1 = 4 * np.multiply(partial_gradient_jbw(X, w, j), np.matmul(ones_d, w.T)).T
        second_term = np.matmul(second_term_1, jb(X, w, j))
        sum_1 += second_term
    sum_1 = sum_1 * lambda1

    third_term = np.multiply(np.multiply(w, w), w) * lambda2 * 4

    sum_2 = 0
    for i in range(X.shape[0]):
        sum_2 += (w[i] * w[i]) - 1
    forth_term = 4 * lambda3 * sum_2 * w

    return first_term + sum_1 + third_term + forth_term


def update_w(X, F, G, w):
    w_not_positive = np.sqrt(w) - learning_rate * partial_gradient_jw(X, F, G, np.sqrt(w))
    return np.multiply(w_not_positive, w_not_positive)


def loss_function(X, F, G, w):
    sum = 0
    for i in range(X.shape[0]):
        sum += np.linalg.norm(X[i] - np.matmul(G[i], F.T)) * w[i]
    return sum/X.shape[0]


def predict(G):
    y_predict = np.zeros((G.shape[0], 1))
    for i in range(G.shape[0]):
        index = np.where(G[i] == 1)
        y_predict[i] = index
    return np.squeeze(y_predict)


def DCKM(X, y):
    clusters = len(np.unique(y))
    # initial random centroids
    index = np.random.choice(X.shape[0], clusters, replace=False)
    F = X[index].T
    w = np.ones((X.shape[0], 1))
    G = update_g(F, X)

    while loss_function(X, F, G, w) > e:
        F = update_f(X, w, G)
        G = update_g(F, X)
        w = update_w(X, F, G, w)
        print(loss_function(X, F, G, w))


    y_predict = predict(G)

    nmi = normalized_mutual_info_score(y, y_predict)
    ari = adjusted_rand_score(y, y_predict)

    return nmi, ari


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
    main()
