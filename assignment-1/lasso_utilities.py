import numpy as np

def norm_squared(x: np.ndarray) -> float:
    return np.power(x, 2).sum(dtype=np.int64)


def gamma_calc(x_norm: float, n: int, lam: float) -> float:
    return (n * lam) / x_norm


def standardize(x: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda col: (col - np.mean(col)) / np.std(col), 0, x)


def soft_threshold(b: float, gamma: float) -> float:
    return np.sign(b) * np.clip((np.abs(b) - gamma), 0, None)


def calc_z_x_inner(
    ind: int,
    y_x_inner: float,
    intercept: float,
    one_x_inner: float,
    weights: np.ndarray,
    x_inners: np.ndarray
) -> float:
    weights_copy = np.squeeze(np.copy(weights))
    weights_copy[ind] = 0
    return y_x_inner - intercept * one_x_inner - sum(np.multiply(weights_copy, x_inners))


def deviance_func(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
    return (((y - b) - np.matmul(x, w))**2).sum(dtype=np.int64)


def score_func(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, lam: float) -> float:
    return (0.5 / y.shape[0]) * deviance_func(x, y, w, b) + lam * np.absolute(b).sum(dtype=np.int64)
