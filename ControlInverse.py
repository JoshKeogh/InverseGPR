import numpy as np
import sklearn.gaussian_process.kernels as kernel
from scipy.spatial.distance import pdist, cdist, squareform


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


def _format(X, Y=None, l_s=None):
    X = np.reshape(X, (-1, int(len(X[0]) / 3), 3))
    Y = np.reshape(Y, (-1, int(len(Y[0]) / 3), 3)) if Y is not None else None
    with np.errstate(divide='ignore'):
        inv_X = 1. / _near_table(X)
        inv_Y = 1. / _near_table(Y) if Y is not None else inv_X

    pair_scale = _pairtype_scale(l_s)

    triu_indices = np.triu_indices_from(pair_scale, k=1)

    return inv_X[:, triu_indices[0], triu_indices[1]], \
           inv_Y[:, triu_indices[0], triu_indices[1]], \
           pair_scale[triu_indices]


class InverseKernel(kernel.RBF):
    def __call__(self, X, Y=None, eval_gradient=False):
        y_none = True if Y is None else False

        X, Y, pair_length = _format(X, Y, aaa)

        if y_none:
            dists = pdist(X / pair_length, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X / pair_length, Y / pair_length,
                          metric='sqeuclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or pair_length.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                             / (pair_length ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K
