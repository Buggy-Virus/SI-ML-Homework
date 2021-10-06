import glmnet_python
from glmnet import glmnet, glmnetPrint
from my_lasso import MyLasso

import numpy as np
import numpy.random as npr

from typing import Callable, Tuple

# DataSet Parameters
DIMDICT = {
    "small": [30, 2],
    "med": [200, 15],
    "big": [2000, 100]
}

# Function And Data Generation
COEF_MAG = 4
INTER_MAG = 4
NOISE_MAG = 2
NOISE_STD = 1.5
X_MAG = 10
X_STD = 4


def r0():
    return npr.rand() - 0.5


def create_func(p: int, coef_s: float, b0_s: float, noise_s: float, std_s: float) -> Callable:
    coefs = coef_s * r0()
    intercept = b0_s * r0()
    return lambda vars: coefs * vars + npr.normal(noise_s * r0(), std_s * r0())


def create_data(func: Callable, n: int, p: int, x_s: float, x_std: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.ndarray
    for i in range(p):
        x = np.hstack(x, npr.normal(x_s * r0(), x_std * r0(), (n, 1)))

    y = np.apply_along_axis(func, 1, x)[np.newaxis].T

    return x, y


def evaluate_mylasso(x: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float, lams: np.ndarray) -> MyLasso:
    return MyLasso().fit(x, y, w, b0, lams=lams)


def evaluate_glmnet(x: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float) -> glmnet:
    fit = glmnet(x=np.copy(x), y=np.copy(y), family='gaussian', weights=w, alpha=1)


def main():
    n, p = DIMDICT["small"]
    func = create_func(p, COEF_MAG, INTER_MAG, NOISE_MAG, NOISE_STD)
    x, y = create_data(func, n, p, X_MAG, X_STD)

    w = npr.rand(p, 1)
    w = (n / np.sum(w)) * w

    glmnet_model = glmnet(x=np.copy(x), y=np.copy(y), family='gaussian', weights=w, alpha=1, nlamba=20)
    mylasso_model = MyLasso().fit(x, y, w, y.mean(), lams=glmnet_model["lambdau"])

    glmnetPrint(glmnet_model)
    mylasso_model.get_results()


if __name__ == "__main__":
    main()