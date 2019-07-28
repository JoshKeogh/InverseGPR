import numpy as np


def near_table(X):
    diff = np.expand_dims(X, axis=2) - X[:, np.newaxis]
    return np.sqrt(np.einsum('...ijk,...ijk->...ij', diff, diff)).squeeze()


def difference_measure_sq(X, l=1.0):
    invR = near_table(X)
    with np.errstate(divide='ignore'):
        invR = np.triu(1/invR, k=1)

    return np.sum((invR[:, np.newaxis] - invR[np.newaxis, :])**2, axis=(2, 3)) / l**2


def inv_kernel(X, l=1.0):
    X = np.reshape(X, (-1, int(len(X[0]) / 3), 3))
    return np.exp((-1/2) * difference_measure_sq(X, l))


def random_sample(path, sample_size=2000):
    data = np.load(path)
    X = data['arr_0']
    y = data['arr_1']

    indices = np.random.choice(y.shape[0], sample_size, replace=False)
    return X[indices], y[indices]


X, y = random_sample("formatted.npz")
print(X)
K = inv_kernel(X, 1.0)
print(K)
