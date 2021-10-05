from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from typing import Callable, Tuple
import numpy as np
import numpy.random as npr


trial_num = 100
training_size = 50
complexity_max = 35
x_mean = 0
x_std = 1
var_num = 3
degree = 4

def generat_curve_function(var_num: int, degree: int, coef_mag: float=2, err_u=0, err_std=1) -> Callable:
    coefs = npr.rand(var_num * degree + 1) * coef_mag
    powers = np.tile(np.arange(degree + 1), var_num)

    func = lambda vars: (
        np.sum(np.multiply(coefs, np.power(np.repeat(vars, degree + 1), powers))) +
        npr.normal(err_u, err_std))

    return func


def main():
    curve_func = generat_curve_function(var_num, degree)

    x = npr.normal(x_mean, x_std, (trial_num, var_num))
    y = np.apply_along_axis(curve_func, 1, x)[np.newaxis].T



if __name__ == "__main__":
    main()