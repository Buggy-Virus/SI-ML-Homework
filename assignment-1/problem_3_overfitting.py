import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

import numpy as np
import numpy.random as npr
import pandas as pd
import warnings

from typing import Callable, Tuple



# Program parameters
TRIALS_N = 100
TRAIN_N = 50
COMPLEXITY_MAX = 35

# Data Parameters
X_MEAN = 5
X_STD = 2

# Function Parameters
VAR_NUM = 1
DEGREE = 9


def generat_curve_function(
    var_num: int,
    degree: int,
    coef_mag: float=2,
    err_u: float=0,
    err_std: float=2
) -> Callable:

    coefs = npr.rand(var_num * (degree + 1)) * coef_mag
    powers = np.tile(np.arange(degree + 1), var_num)

    func = lambda vars: (
        np.sum(
            np.multiply(
                coefs,
                np.power(np.repeat(vars, degree + 1), powers)
            )
        )
        + npr.normal(err_u, err_std))

    return func


def create_data(
    func: Callable,
    n: int,
    p: int,
    x_s: float,
    x_std: float
) -> Tuple[np.ndarray, np.ndarray]:

    x = None
    for i in range(p):
        new_col = npr.normal(x_s * (npr.rand() - 0.5), x_std * npr.rand(), (n, 1))
        if x is not None:
            x = np.hstack((x, new_col))
        else:
            x = new_col

    y = np.apply_along_axis(func, 1, x)[np.newaxis].T

    return x, y


def get_curves(
    curve_func: Callable,
    max_iterations: int,
    max_complexity: int,
    training_size: int,
    var_num: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    x_coords = np.arange(max_complexity + 1)
    blue_lines = None
    red_lines = None

    for _ in range(max_iterations):
        blue_line = None
        red_line = None

        for i in range(max_complexity + 1):
            x, y = create_data(curve_func, 2 * training_size, var_num, X_MEAN, X_STD)

            x_training, x_test = np.vsplit(x, 2)
            y_training, y_test = np.vsplit(y, 2)

            model = Pipeline([('poly', PolynomialFeatures(degree=i)),
                              ('linear', Lasso(alpha=0.25))])
                            # ('linear', LinearRegression(fit_intercept=False))])
            model.fit(x_training, y_training)

            y_trained = model.predict(x_training)
            train_mse = mean_squared_error(y_training, y_trained)
            if blue_line is None:
                blue_line = np.array([train_mse])
            else:
                blue_line = np.append(blue_line, train_mse)

            y_pred = model.predict(x_test)
            expec_mse = mean_squared_error(y_test, y_pred)
            if red_line is None:
                red_line = np.array([expec_mse])
            else:
                red_line = np.append(red_line, expec_mse)

        if blue_lines is not None:
            blue_lines = np.vstack((blue_lines, blue_line))
            red_lines = np.vstack((red_lines, red_line))
        else:
            blue_lines = blue_line
            red_lines = red_line

    blue_lines = np.clip(blue_lines, 0, np.max(blue_lines[:,0]))
    red_lines = np.clip(red_lines, 0, np.max(red_lines[:,max_complexity]))

    bright_blue = np.mean(blue_lines, axis=0)
    bright_red = np.mean(red_lines, axis=0)

    return blue_lines, red_lines, bright_blue, bright_red, x_coords


def plot_curves(
    b_lines: np.ndarray, r_lines: np.ndarray,
    b_blue: np.ndarray, b_red: np.ndarray,
    x: np.ndarray, iterations: int):

    b_c = (31 / 256, 33 / 256, 204 / 256)
    r_c = (204 / 256, 31 / 256, 31 / 256)
    basic_b_c = (80 / 256, 82 / 256, 217 / 256, 0.5)
    basic_r_c = (217 / 256, 80 / 256, 80 / 256, 0.5)

    b_logs = np.log(b_lines)
    r_logs = np.log(r_lines)

    for i in range(iterations):
        plt.plot(x, b_logs[i, :], '-', 'LineWidth', 0.65, color=basic_b_c)
        plt.plot(x, r_logs[i, :], '-', 'LineWidth', 0.65, color=basic_r_c)

    b_b_log = np.log(b_blue)
    b_r_log = np.log(b_red)

    plt.plot(x, b_b_log, '-', 'LineWidth', 0.85, label='Exp. T Err', color=b_c)
    plt.plot(x, b_r_log, '-', 'LineWidth', 0.85, label='Est. Err', color=r_c)

    plt.xlabel("Model Complexity (df)")
    plt.ylabel("Log(Prediction Error)")
    plt.title("Test Error vs. Complexity")

    old_min, old_max = plt.ylim()
    plt.ylim(old_max * (np.min(b_b_log) / np.min(b_logs)) / np.max(r_logs), old_max * np.max(b_r_log) / np.max(r_logs))

    plt.show()


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    func = generat_curve_function(VAR_NUM, DEGREE)
    b_lines, r_lines, b_blue, b_red, x = get_curves(func, TRIALS_N, COMPLEXITY_MAX, TRAIN_N, VAR_NUM) # long line
    plot_curves(b_lines, r_lines, b_blue, b_red, x, TRIALS_N)

    c = COMPLEXITY_MAX
    desc_data = [ (b_blue[0], b_red[0], np.std(b_lines[:,0]), np.std(r_lines[:,0])),
             (b_blue[c], b_red[c], np.std(b_lines[:,c]), np.std(r_lines[:,c])) ]
    desc_df = pd.DataFrame(desc_data, columns = ['TE u' , 'CTE u', 'TE std' , 'CTE std'], index=[0, c]) # long line
    print(desc_df)


if __name__ == "__main__":
    main()