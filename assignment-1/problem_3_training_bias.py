import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

import numpy as np
import numpy.random as npr
import pandas as pd

from typing import Callable, Tuple

# Program parameters
TRIALS_N = 100
TRAIN_N = 50
COMPLEXITY_MAX = 35

# Data Parameters
X_MEAN = 0
X_STD = 1

# Function Parameters
VAR_NUM = 3
DEGREE = 4


def generat_curve_function(
    var_num: int,
    degree: int,
    coef_mag: float=2,
    err_u: float=0,
    err_std: float=1
) -> Callable:

    coefs = npr.rand(var_num * degree + 1) * coef_mag
    powers = np.tile(np.arange(degree + 1), var_num)

    func = lambda vars: (
        np.sum(
            np.multiply(
                coefs,
                np.power(np.repeat(vars, degree + 1), powers)
            )
        ) + npr.normal(err_u, err_std))

    return func


def get_curves(
    curve_func: Callable,
    max_iterations: int,
    max_complexity: int,
    training_size: int,
    var_num: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    x_coords = np.arange(max_complexity + 1)
    blue_lines = np.array([])
    red_lines = np.array([])

    for _ in range(max_iterations):
        blue_line = np.ndarray([])
        red_line = np.ndarray([])

        for i in range(max_complexity + 1):
            x = npr.normal(X_MEAN, X_STD, (2 * training_size, var_num))
            y = np.apply_along_axis(curve_func, 1, x)[np.newaxis].T

            x_training, x_test = np.vsplit(x, 2)
            y_training, y_test = np.vsplit(y, 2)

            model = Pipeline([('poly', PolynomialFeatures(degree=i)),
                            ('linear', Lasso(alpha=0.25))])
            model.fit(x_training, y_training)

            y_trained = model.predict(x_training)
            train_mse = mean_squared_error(y_training, y_trained)
            blue_line = np.append(blue_line, train_mse)

            y_pred = model.predict(x_test)
            expec_mse = mean_squared_error(y_test, y_pred)
            red_line = np.append(red_line["x"], expec_mse)

        np.vstack((blue_lines, blue_line))
        np.vstack((red_lines, red_line))

    bright_blue = np.mean(blue_lines, axis=0)
    bright_red = np.mean(red_lines, axis=0)

    return blue_lines, red_lines, bright_blue, bright_red, x_coords


def plot_curves(
    b_lines: np.ndarray, r_lines: np.ndarray,
    b_blue: np.ndarray, b_red: np.ndarray,
    x: np.ndarray, iterations: int):

    b_c = (31, 33, 204)
    r_c = (204, 31, 31)
    basic_b_c = (80, 82, 217, 0.5)
    basic_r_c = (217, 80, 80, 0.5)

    for i in range(iterations):
        plt.plot(x, b_lines[i, :], '-', 'LineWidth', 0.65, color=basic_b_c)
        plt.plot(x, r_lines[i, :], '-', 'LineWidth', 0.65, color=basic_r_c)

    plt.plot(x, b_blue, '-', 'LineWidth', 0.85, label='Exp. T Err', color=b_c)
    plt.plot(x, b_red, '-', 'LineWidth', 0.85, label='Est. Err', color=r_c)

    plt.xlabel("Model Complexity (df)")
    plt.ylabel("Prediction Error")
    plt.title("Sine and Cosine functions")
    plt.legend("Impact of Overfitting on Test Error vs. Conditional Test Error")

    plt.show()


def main():
    func = generat_curve_function(VAR_NUM, DEGREE)
    b_lines, r_lines, b_blue, b_red, x = get_curves(func, TRIALS_N, COMPLEXITY_MAX, TRAIN_N) # long line
    plot_curves(b_lines, r_lines, b_blue, b_red, x, TRIALS_N)

    c = COMPLEXITY_MAX
    desc_data = [ (b_blue[0], b_red[0], np.std(b_lines[:,0]), np.std(r_lines[:,0])),
             (b_blue[c], b_red[c], np.std(b_lines[:,c]), np.std(r_lines[:,c])) ]
    desc_df = pd.DataFrame(desc_data, columns = ['TE u' , 'CTE u', 'TE std' , 'CTE std'], index=[0, c])
    print(desc_df)


if __name__ == "__main__":
    main()