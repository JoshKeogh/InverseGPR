import numpy as np
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel


def near_table(X):
    diff = np.expand_dims(X, axis=2) - X[:, np.newaxis]
    return np.sqrt(np.einsum('...ijk,...ijk->...ij', diff, diff)).squeeze()


def difference_measure(X, l=1.0):
    invR = near_table(X)
    with np.errstate(divide='ignore'):
        invR = np.triu(1 / invR, k=1)
    return np.sum((invR[:, np.newaxis] - invR[np.newaxis, :]) ** 2, axis=(2, 3)) / l**2


class InverseKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def _check_length_scale(X, length_scale):
        length_scale = np.squeeze(length_scale).astype(float)
        if np.ndim(length_scale) > 1:
            raise ValueError("length_scale cannot be of dimension greater than 1")
        if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
            raise ValueError("Anisotropic kernel must have the same number of "
                             "dimensions as data (%d!=%d)"
                             % (length_scale.shape[0], X.shape[1]))
        return length_scale

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.reshape(X, (-1, int(len(X[0]) / 3), 3))
        return np.exp((-1/2) * difference_measure(X, self.length_scale))

