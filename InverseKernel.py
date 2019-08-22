import numpy as np
import sklearn.gaussian_process.kernels as kernel
from scipy.spatial.distance import squareform


def _near_table(X):
    """Compute the pairwise difference between each atom in the samples of X.
    r_ij = sqrt(sigma_d (x_id - x_jd)^2)
    Parameters
    ----------
    X   : numpy.ndarray, shape (n_samples_X, n_atoms, n_coordinates)
        Co-ordinates of the atoms in each sample.
    Returns
    -------
    r_ij    : numpy.ndarray, shape (n_samples_X, n_atoms, n_atoms)
            The Euclidean distance between each of the atoms within a sample
    """
    diff = X[:, :, np.newaxis] - X[:, np.newaxis]
    return np.sqrt(np.einsum('...jk,...jk->...j', diff, diff)).squeeze()


def _pairtype_scale(length_scale):
    a_0 = np.array(length_scale)
    return np.ones((len(a_0), len(a_0))) * a_0[:, np.newaxis] + a_0[:, np.newaxis].T


def _difference_measure_sq(length_scale, x_left, x_right=None):
    """Computes the square of the difference measure, which is defined as
    D(x, x')^2 = sigma_i sigma_j (1/r_ij(x) - 1/r_ij(x'))^2 / l^2
    where r_ij = sqrt(sigma_d (x_id - x_jd)^2)
    Used in calculating a squared-exponential covariance function.
    Parameters
    ----------
    x_left  : numpy.ndarray, shape (n_samples_X1, n_atoms, n_atoms)
            Left argument of the covariance function, x.
    x_right : numpy.ndarray, shape (n_samples_X2, n_atoms, n_atoms), (optional, default=None)
            Right argument of the covariance function, x'.
            If None, x_left is used for right argument.
    l   : float (optional, default=1.0)
        The length scale of the kernel.
    Returns
    -------
    D   : numpy.ndarray, shape (n_samples_right, n_samples_left)
        Difference measure using inverted inter-atomic distances
    """
    inv = lambda x: np.triu(1. / np.multiply(_near_table(x), length_scale), k=1)
    with np.errstate(divide='ignore'):
        invR_left = inv(x_left)
        invR_right = inv(x_right) if x_right is not None else invR_left

    return np.sum((invR_left[:, np.newaxis] - invR_right[np.newaxis, :]) ** 2, axis=(2, 3))


def _gram_matrix(x_left, x_right=None, a_0=None):
    """Generate the squared-exponential kernel given by
    k(x_i, x_j) = exp(-1/2 * D(x_left / l, x_right / l)^2)
    where D is the difference measure using inverse inter-atomic distances.
    Parameters
    ----------
    x_left  : numpy.ndarray, shape (n_samples_X1, n_atoms * n_coordinates)
            Left argument of the covariance function.
    x_right : numpy.ndarray, shape (n_samples_X2, n_atoms * n_coordinates), (optional, default=None)
            Right argument of the covariance function.
            If None, x_left is used for right argument.
    l   : float (optional, default=1.0)
        The length scale of the kernel.
    Returns
    -------
    K(x, x')    : numpy.ndarray, shape (n_samples_right, n_samples_left)
                The covariance matrix for the kernel
    """
    if a_0 is None:
        a_0 = [1.]
    length_scale = _pairtype_scale(a_0)
    
    format = lambda x: np.reshape(x, (-1, int(len(x[0]) / 3), 3))
    x_left = format(x_left)
    x_right = format(x_right) if x_right is not None else None

    diff = _difference_measure_sq(length_scale, x_left, x_right)
    return np.exp(-0.5 * diff)


class InverseKernel(kernel.StationaryKernelMixin, kernel.NormalizedKernelMixin, kernel.Kernel):
    """A squared-exponential kernel whose covariance function uses the reciprocal
    of distances in generating a difference measure.
    The kernel is given by:
    k(x_i, x_j) = exp(-1/2 D(x_i / length_scale, x_j / length_scale)^2)
    where D(x, x')^2 = sigma_i sigma_j (1/r_ij(x) - 1/r_ij(x'))^2 / l^2
    and r_ij = sqrt(sigma_d (x_id - x_jd)^2)
    Parameters
    ----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """

    a_0 = [0.77, 0.32, 0.32, 0.32, 0.32]

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y)
        Parameters
        ----------
        X : array, shape (n_samples_X, n_atoms * n_coodinates)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_atoms * n_coodinates), (optional, default=None)
            Right argument of the returned kernel k(X, Y).
            If None, k(X, X) if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        """
        if Y is None:
            K = _gram_matrix(X, a_0=self.a_0)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            K = _gram_matrix(X, Y, a_0=self.a_0)
            
        if eval_gradient:
            l = _pairtype_scale(self.a_0)
            X = np.reshape(X, (-1, int(len(X[0]) / 3), 3))
            D = _difference_measure_sq(l, X)
            
            K_gradient = D[:, :, np.newaxis] * K[..., np.newaxis]
            
            return K, K_gradient

        return K
