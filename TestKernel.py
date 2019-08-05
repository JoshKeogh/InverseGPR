import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from InverseKernel import InverseKernel


def random_sample(path, sample_size=200):
    data = np.load(path)
    x = data['arr_0']
    y = data['arr_1']
    x = StandardScaler().fit_transform(x)
    indices = np.random.choice(y.shape[0], sample_size, replace=False)
    return x[indices], y[indices]


def display(y_actual, y_prediction, out_path=None):
    samples = len(y_prediction)

    plt.scatter(y_actual, y_prediction, color='green', alpha=0.3)
    plt.suptitle("Predictions for " + str(samples) + " samples using inverse distances")
    plt.xlabel("y measured value")
    plt.ylabel("y prediction")
    plt.show()
    if out_path is not None:
        fig = plt.figure()
        fig.savefig(out_path + str(samples) + ".png")

    print(samples, " samples")
    print("R^2\t: ", r2_score(y_actual, y_prediction))
    print("MAE\t: ", mean_absolute_error(y_actual, y_prediction))
    print("MSE\t: ", mean_squared_error(y_actual, y_prediction))
    if out_path is not None:
        with open(out_path + "errors.txt", "a") as file:
            file.write(str(samples) + " samples")
            file.write("\nR^2\t" + str(r2_score(y_actual, y_prediction)))
            file.write("\nMAE\t" + str(mean_absolute_error(y_actual, y_prediction)))
            file.write("\nMSE\t" + str(mean_squared_error(y_actual, y_prediction)) + "\n")


def evaluate_data(in_path, alpha=1e-5):
    for i in range(2000, 3001, 100):
        x, y = random_sample(in_path, i)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        kernel = InverseKernel()
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
        gp.fit(x_train, y_train)

        y_pred = gp.predict(x_test)
        display(y_test, y_pred)

'''
    i = int(1e4)
    x, y = random_sample(in_path, i)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
    gp.fit(x_train, y_train)
    y_pred = gp.predict(x_test)

    display(y_test, y_pred)
'''


evaluate_data("data/CH4-Energies.npz", 5e-5)
