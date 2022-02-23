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
    X_train_oh1, y_train_oh1 = read_dataset('data\OH1.mat')
    oh1_nmi, oh1_ari = train_model(X_train_oh1, y_train_oh1)

    X_train_oh2, y_train_oh2 = read_dataset('data\OH2.mat')
    oh2_nmi, oh2_ari = train_model(X_train_oh2, y_train_oh2)

    X_train_oh3, y_train_oh3 = read_dataset('data\OH3.mat')
    oh3_nmi, oh3_ari = train_model(X_train_oh3, y_train_oh3)

    table = tabulate([['OH1', oh1_nmi, oh1_ari], ['OH2', oh2_nmi, oh2_ari], ['OH3', oh3_nmi, oh3_ari]],
                     headers=['Dataset', 'NMI', 'ARI'])

    print(table)

if __name__ == '__main__':
    main()
