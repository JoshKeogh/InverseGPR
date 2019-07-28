import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def random_sample(path, sample_size):
    data = np.load(path)
    X = data['arr_0']
    y = data['arr_1']

    indices = np.random.choice(y.shape[0], sample_size, replace=False)
    return X[indices], y[indices]


X, y = random_sample("formatted.npz", 2000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

kernel = RBF()
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(X_train, y_train)

print(gp.score(X_test, y_test))
