import os

import numpy
import scipy.io
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from tabulate import tabulate


def read_dataset(path):
    mat = scipy.io.loadmat(path)
    return mat['X_train'], mat['label'].reshape(mat['label'].shape[0])


def train_model(X, y):
    clusters = len(numpy.unique(y))
    kmeans = KMeans(random_state=0, n_clusters=clusters)
    kmeans.fit(X)
    y_predict = kmeans.predict(X)
    nmi = normalized_mutual_info_score(y, y_predict)
    ari = adjusted_rand_score(y, y_predict)

    return nmi, ari


def main():
    datasets = os.listdir('data/')
    metrics = []
    for dataset in datasets:
        X_train, y_train = read_dataset(f'data\{dataset}')
        nmi, ari = train_model(X_train, y_train)
        metrics += [[dataset, nmi, ari]]

    table = tabulate(metrics, headers=['Dataset', 'NMI', 'ARI'])
    print(table)


if __name__ == '__main__':
    main()
