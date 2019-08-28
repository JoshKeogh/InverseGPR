import numpy as np
import sklearn.gaussian_process.kernels as kernel
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.preprocessing import StandardScaler


def _near_table(X):
    diff = X[:, :, np.newaxis] - X[:, np.newaxis]
    return np.sqrt(np.einsum('...jk,...jk->...j', diff, diff)).squeeze()


def _pairtype_scale(a_0=None):
    if a_0 is None:
        a_0 = [1.]
    a_0 = np.array(a_0)
    a_0 = np.ones((len(a_0), len(a_0))) * a_0[:, np.newaxis] + a_0[:, np.newaxis].T
    return a_0


aaa = [0.77, 0.32, 0.32, 0.32, 0.32]


def _difference_measure_sq(length_scale, x_left, x_right=None):
    inv = lambda x: np.triu(1. / np.multiply(x, length_scale), k=1)
    with np.errstate(divide='ignore'):
        invR_left = inv(x_left)
        invR_right = inv(x_right) if x_right is not None else invR_left

    return np.sum((invR_left[:, np.newaxis] - invR_right[np.newaxis, :]) ** 2, axis=(2, 3))


def _format(X, Y=None, l_s=None):
    X = np.reshape(X, (-1, int(len(X[0]) / 3), 3))
    Y = np.reshape(Y, (-1, int(len(Y[0]) / 3), 3)) if Y is not None else None
    with np.errstate(divide='ignore'):
        X = 1. / _near_table(X)
        Y = 1. / _near_table(Y) if Y is not None else None

    pair_scale = _pairtype_scale(l_s)

    #triu_indices = np.triu_indices_from(pair_scale, k=1)

    return X, Y, pair_scale  #[:, triu_indices[0], triu_indices[1]], \
           #Y,[:, triu_indices[0], triu_indices[1]] if Y is not None else None, \
           #pair_scale[triu_indices]


class InverseKernel(kernel.RBF):
    scaler = StandardScaler()

    def __call__(self, X, Y=None, eval_gradient=False):
        X, Y, pair_scale = _format(X, Y, aaa)

        dists = _difference_measure_sq(pair_scale, X, Y)
        K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or pair_scale.shape[0] == 1:
                K_gradient = \
                    (K * dists)[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                             / (pair_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, -0.5 * K_gradient
        else:
            return K
