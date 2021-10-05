import numpy as np

def norm_squared(x: np.ndarray) -> float:
    return np.sum(x**2)


def gamma_calc(n: int, lam: float, x_norm: float) -> float:
    return (n * lam) / x_norm


def standardize(x: np.ndarray) -> np.ndarray:
    return x.apply_along_axis(lambda col: (col - np.mean(col)) / np.std(col), 0, x)


def soft_threshold(b: float, gamma: float) -> float:
    return np.sign(b) * max((abs(b) - gamma), 0)


def calc_z_x_inner(
        ind: int, y_x_inner: float, intercept: float, one_x_inner: float,
        weights: np.ndarray, x_inners: np.ndarray) -> float:
    weights_copy = np.squeeze(np.copy(weights))
    weights_copy[ind] = 0
    return y_x_inner - intercept * one_x_inner - sum(np.multiply(weights_copy, x_inners))


def deviance_func(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
    return np.sum(((y - b) - np.matmul(x, w))**2)


def score_func(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, lam: float) -> float:
    return (0.5 / y.shape[0]) * deviance_func(x, y, w, b) + lam * np.sum(np.absolute(b))