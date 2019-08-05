import numpy as np
import sklearn.gaussian_process.kernels as kernel


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
    diff = np.expand_dims(X, axis=2) - X[:, np.newaxis]
    return np.sqrt(np.einsum('...jk,...jk->...j', diff, diff)).squeeze()


def _difference_measure_sq(x_left, x_right=None, l=1.0):
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
    inv = lambda x: np.triu(1 / _near_table(x), k=1)
    with np.errstate(divide='ignore'):
        if x_right is None:
            invR = inv(x_left)
            return np.sum((invR[:, np.newaxis] - invR[np.newaxis, :])**2, axis=(2, 3))
        else:
            return np.sum((inv(x_left)[:, np.newaxis] - inv(x_right)[np.newaxis, :])**2, axis=(2, 3)) / l**2


def _gram_matrix(x_left, x_right=None, l=1.0):
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
                Difference measure using inverted inter-atomic distances
    """
    format = lambda x: np.reshape(x, (-1, int(len(x[0]) / 3), 3))
    x1 = format(x_left)
    x2 = None if x_right is None else format(x_right)
    diff = _difference_measure_sq(x1, x2, l)
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
            K = _gram_matrix(X, l=self.length_scale)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            K = _gram_matrix(X, Y, l=self.length_scale)

        print("K.shape", K.shape)
        if eval_gradient:
            raise NotImplementedError("eval_gradient not implemented in __call__")

        return K
