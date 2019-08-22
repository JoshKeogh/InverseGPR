import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
import seaborn as sbn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def random_sample(path, sample_size=100, seed=None):
    data = np.load(path)
    x = data['arr_0']
    y = data['arr_1']
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(y.shape[0], sample_size, replace=False)
    return x[indices], y[indices]


def display(y_actual, y_prediction, baseline_guess, kernel_type, out_path=None, index="default", fit_line=True):
    samples = len(y_prediction)

    fig = plt.figure()
    plt.scatter(y_actual, y_prediction, color='green', alpha=0.3)
    if fit_line:
        plt.plot(np.unique(y_actual), np.poly1d(np.polyfit(y_actual, y_prediction, 1))(np.unique(y_actual)), color='k')
    fig.suptitle("Predictions for " + str(samples) + " samples using " + kernel_type)

    '''

    sbn.regplot(y_actual, y_prediction)
    plt.title("Predictions for " + str(samples) + " samples using " + kernel_type)
    '''
    plt.xlabel("y measured value")
    plt.ylabel("y prediction")
    if out_path is not None:
        fig.savefig(out_path + index + ".png")
    plt.show()

    print('The baseline guess is a score of %0.2f' % baseline_guess)
    print("Baseline Performance on the test set: MAE = %0.4f" % np.mean(abs(y_actual - baseline_guess)))
    print(samples, " predictions")
    print("R^2\t: ", r2_score(y_actual, y_prediction))
    print("MAE\t: ", mean_absolute_error(y_actual, y_prediction))
    print("MSE\t: ", mean_squared_error(y_actual, y_prediction))
    if out_path is not None:
        with open(out_path + "errors.txt", "a") as file:
            file.write('The baseline guess is a score of %0.2f' % baseline_guess)
            file.write("\nBaseline Performance on the test set: MAE = %0.4f" % np.mean(abs(y_actual - baseline_guess)))
            file.write("\n" + str(samples) + " samples")
            file.write("\nR^2\t" + str(r2_score(y_actual, y_prediction)))
            file.write("\nMAE\t" + str(mean_absolute_error(y_actual, y_prediction)))
            file.write("\nMSE\t" + str(mean_squared_error(y_actual, y_prediction)) + "\n")


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


def pairtype_scale(a_0):
    a_0 = np.array(a_0)
    a_0 = np.ones((len(a_0), len(a_0))) * a_0[:, np.newaxis] + a_0[:, np.newaxis].T
    triu_indices = np.triu_indices_from(a_0, k=1)
    return a_0[triu_indices]


def dostuff():
    a_0 = [0.77, 0.32, 0.32, 0.32, 0.32]
    length_scale = pairtype_scale(a_0)
    for i in range(0, 5):
        x, y = random_sample("data/CH4-Energies.npz", 200, i)

        x = np.reshape(x, (-1, int(len(x[0]) / 3), 3))
        with np.errstate(divide='ignore'):
            inv_r = 1. / _near_table(x)

        inv_r_train, inv_r_test, y_train, y_test = train_test_split(inv_r[:, triu_indices[0], triu_indices[1]], y, test_size=0.2, random_state=0)

        triu_indices = np.triu_indices_from(inv_r_train[0], k=1)

        kernel = WhiteKernel(noise_level=1e-3) + RBF(length_scale=length_scale)

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp.fit(inv_r_train, y_train)

        y_pred = gp.predict(inv_r_test)

        display(y_test, y_pred, np.median(y), kernel_type="Inverted Distances with RBF")#, out_path='data/', index=str(i))


dostuff()
