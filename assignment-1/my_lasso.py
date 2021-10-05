from typing import Tuple
import numpy as np
import pandas as pd

import lasso_utilities as lu

pd.set_option("display.max_rows", None, "display.max_columns", None)

class MyLasso:

    def __init__(self):
        # Input
        self.x = None # n x p, each row is a data point, n predictors
        self.y = None # n x 1,

        self.n = None
        self.p = None

        self.initial_w = None # p x 1
        self.initial_b0 = None

        # Results
        self.final_score = float("inf")
        self.final_w = None
        self.final_b0 = None

        # Precomputation variables
        self.y_mean = None
        self.x_means = None # p
        self.x_norms_sqrd = None # p
        self.x_y_inners = None # p
        self.one_x_inners = None # p
        self.x_inners = None # p x p

        # cross validation variables
        self.results = {"score": [], "dev": [], "%%dev": [],
                        "df": [], "lambda": [], "w": [], "b0": []}
        self.null_dev = None


    def initialize(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray):
        self.n, self.p = x.shape[0], x.shape[1]
        if self.n != y.shape[0] or y.shape[1] != 1:
            raise ValueError(
                f"Incompatible input data dimensions: " +
                "x={self.n}x{self.p}, y={y.shape[0]}x{y.shape[0]}")

        self.x = lu.standardize(x)
        self.y = y, self.initial_w, self.initial_b0 = w, b


    def precomputation(self):
        self.y_mean = self.y.mean()
        self.x_means = np.mean(self.x, axis=0) # p
        self.x_norms_sqrd = np.apply_along_axis(lu.norm_squared, 0, self.x) # p
        self.x_y_inners = np.squeeze(np.matmul(np.rot90(self.y, 1), self.x)) # 1 x p -> p
        self.one_x_inners = np.squeeze(np.matmul(np.ones((1, self.n)), self.x)) # 1 x p -> p
        self.x_inners = np.matmul(np.rot90(np.flipud(self.x), 3), self.x) # p x p
        self.null_dev = lu.deviance_func(0, self.y, 0, self.y_mean)


    def calc_gammas(self, lam) -> np.ndarray:
        return np.apply_along_axis(lu.gamma_calc, 0, self.x_norms_sqrd, lam=lam) # p


    def gen_lambdas(self, nlam: int) -> np.ndarray:
        # looked at the glmnet code to see what they use
        if self.n < self.p:
            lambda_min = 0.01
        else:
            lambda_min = 1e-4

        # minimum lambda such that all coefficients are zero
        lambda_max = (1 / self.n) * np.max(np.abs(np.squeeze(np.matmul(np.transpose(self.x), self.y))))

        # based on the glmnet docs for how they choose lambdas
        step_size = (np.log(lambda_max) - np.log(lambda_min)) / (nlam - 1)
        return np.exp((np.ndarray(range(20)) * step_size) + np.log(lambda_min))


    def calc_new_values(self, weights: np.ndarray, b0: float,
            gammas: np.ndarray, ind: int) -> Tuple[float, float]:
        new_b0 = self.y_mean - sum(np.multiply(self.initial_weights, self.x_means))

        gamma = gammas[ind]
        z_x_inner = lu.calc_z_x_inner(
            ind, self.x_y_inners, b0,
            self.one_x_inners, weights,
            self.x_inners[:, ind])

        b_hat = z_x_inner / self.x_norms_sqrd[ind]
        new_coord_weight = lu.soft_threshold(b_hat, gamma)

        return new_coord_weight, new_b0


    def gen_model(self, lam: float, total_iters: int):
        gamma = self.calc_gammas(lam)
        w = np.copy(self.initial_w)
        b0 = self.initial_b0

        run_book = {"score": [], "beta": [], "b0": []}
        iters = 0
        score = float("inf")
        new_score = lu.score_func(self.x, self.y, w, b0, lam)
        while new_score < score and iters < total_iters:
            iters += 1
            new_entry = {"score": new_score, "beta": w, "b0": b0}
            for key, value in new_entry.items():
                run_book[key].append(value)

            for i in range(self.p):
                w[i,0], b0 = self.calc_new_values(w, b0, gamma, i)

            score = new_score
            new_score = lu.score_func(self.x, self.y, w, b0, lam)

            if iters == total_iters:
                print(f"Max number of iterations reached for lam: {lam}")
                print(pd.DataFrame.from_dict(run_book).tail(50))

        return new_score, w, b0


    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray,
            b0: float, thresh: float, iters: int=1000,
            lams: np.ndarray=None, nlam: int=20):
        self.initialize(x, y, w, b)
        self.precomputation()

        if lams == None:
            lams = self.calc_lambdas(nlam)

        for lam in lams:
            model_score, model_w, model_b0 = self.gen_model(w, b0, lam)
            model_dev = lu.deviance_func(self.x, self.y, model_w, model_b0)
            model_pde = 1 - (model_dev / self.null_dev)
            model_df = np.count_nonzero(model_w)

            if model_score < self.final_score:
                self.finale_score, self.final_w, self.final_b0 = model_score, model_w, model_b0

            model_entry = {"score": model_score, "dev": model_dev, "%%dev": model_pde,
                           "df": model_df, "lambda": model_entry, "w": model_w, "b0": model_b0}
            for key, value in model_entry.items():
                self.results[key].append(value)

    def get_results(self, print=True):
        if print:
            print("MyLasso Results")
            print("Best Score: {self.final_score}")
            print_cols = ["score", "dev", "%%dev", "df", "lambda"]
            print(pd.DataFrame.from_dict(self.results)[print_cols])

        return self.final_score, self.final_w, self.final_b0, self.results