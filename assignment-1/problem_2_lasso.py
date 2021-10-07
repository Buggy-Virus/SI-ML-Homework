import glmnet_python
from glmnet import glmnet

from my_lasso import MyLasso

import numpy as np
import numpy.random as npr
import pandas as pd

from typing import Callable, Tuple

# Function And Data Generation
COEF_MAG = 4
INTER_MAG = 4
NOISE_MAG = 2
NOISE_STD = 1.5
X_MAG = 10
X_STD = 1


def r0():
    return npr.rand() - 0.5


def create_func(p: int, coef_s: float, b0_s: float, noise_s: float, std_s: float) -> Callable:
    coefs = coef_s * r0()
    intercept = b0_s * r0()
    return lambda vars: coefs * vars + npr.normal(noise_s * r0(), std_s * r0())


def given_func(point: np.ndarray) -> float:
    result = 3 * point[0] - 17 * point[1] + 5 * point[2] + npr.randn()
    return result


def create_data(func: Callable, n: int, p: int, x_s: float, x_std: float) -> Tuple[np.ndarray, np.ndarray]:
    x = None
    for i in range(p):
        new_col = npr.normal(x_s * (npr.rand() - 0.5), x_std * npr.rand(), (n, 1))
        if x is not None:
            x = np.hstack((x, new_col))
        else:
            x = new_col

    y = np.apply_along_axis(func, 1, x)[np.newaxis].T

    return x, y


def evaluate_mylasso(x: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float, lams: np.ndarray) -> MyLasso:
    return MyLasso().fit(x, y, w, b0, lams=lams)


def evaluate_glmnet(x: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float) -> glmnet:
    fit = glmnet(x=np.copy(x), y=np.copy(y), family='gaussian', weights=w, alpha=1)


def main():
    np.random.seed(50)

    n = 2000
    p = 3

    # func = create_func(p, COEF_MAG, INTER_MAG, NOISE_MAG, NOISE_STD)
    x, y = create_data(given_func, n, p, X_MAG, X_STD)
    w = np.ones((p, 1))

    glmnet_model = glmnet(x=np.copy(x), y=np.copy(y), family ='gaussian', alpha=1, nlambda=40)
    mylasso_model = MyLasso()
    mylasso_model.fit(x, y, w, y.mean(), lams=glmnet_model["lambdau"])

    betas = pd.DataFrame(glmnet_model["beta"].T, columns=["x1", "x2", "x3"])
    df = pd.DataFrame(glmnet_model["df"], columns = ['df'])
    lam = pd.DataFrame(glmnet_model["lambdau"], columns = ['lambdau'])
    dev = pd.DataFrame(glmnet_model["dev"], columns = ['dev'])
    nulldev = pd.DataFrame(glmnet_model["nulldev"], columns = ['nulldev'])

    print(dev.join([nulldev, df, lam, betas]))

    mylasso_model.get_results()

    print("here")


if __name__ == "__main__":
    main()
