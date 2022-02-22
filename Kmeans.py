import scipy.io


def read_dataset(path):
    mat = scipy.io.loadmat(path)
    return mat['X_train'], mat['label']


X_train_oh1, y_train_oh1 = read_dataset('data\OH1.mat')
X_train_oh2, y_train_oh2 = read_dataset('data\OH2.mat')
X_train_oh3, y_train_oh3 = read_dataset('data\OH3.mat')
