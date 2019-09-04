import Display as disp
import TestKernel as data
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import seaborn as sbn


def near_table(X):
    diff = X[:, :, np.newaxis] - X[:, np.newaxis]
    return np.sqrt(np.einsum('...jk,...jk->...j', diff, diff)).squeeze()


def combination_rule(a_0):
    a_0 = np.array(a_0)
    if a_0.size > 1:
        a_0 = np.ones((len(a_0), len(a_0))) * a_0[:, np.newaxis] + a_0[:, np.newaxis].T
    return a_0


def coordination_number(Rij, n_atoms, index, CN_width=0.1):
    N = Rij.shape[0]
    CN = np.zeros((N, n_atoms))

    for i in range(n_atoms):
        for k, (ip, jp) in enumerate(index):
            if ip == i or jp == i:
                CN[:, i] += np.exp(-(Rij[:, k] - 1.) ** 2 / CN_width ** 2)

    return CN


for i in range(0, 10):
    XYZ, y = data.random_sample('data/CH4-QC.npz', 200, i)
    n_atoms = len(XYZ[0]) // 3

    length_scale = [0.77, 0.32, 0.32, 0.32, 0.32]
    paired_scale = combination_rule(length_scale)
    triu_indices = np.triu_indices_from(paired_scale, k=1)

    XYZ = np.reshape(XYZ, (-1, n_atoms, 3))
    XYZ = near_table(XYZ)[:, triu_indices[0], triu_indices[1]]
    with np.errstate(divide='ignore'):
        coord_num = coordination_number(XYZ / paired_scale[triu_indices], n_atoms,
                                        zip(triu_indices[0], triu_indices[1]))
        XYZ = 1. / XYZ

    x_red = np.hstack((XYZ, coord_num))

    x_train, x_test, y_train, y_test = train_test_split(x_red, y, test_size=0.2, random_state=0)

    scaler = RobustScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    kernel = WhiteKernel(noise_level=1e-5) + RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(x_train, y_train)

    y_pred = gp.predict(x_test)

    disp.display(y_test, y_pred, np.median(y), kernel_type="RBF Kernel")
