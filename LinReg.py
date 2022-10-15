import numpy as np
import matplotlib.pyplot as plt
from os import path


class LinReg:
    def __init__(self, learning_rate: float = 0.001, eps: float = 1e-6, theta=None) -> None:
        self.lr = learning_rate
        self.eps = eps
        self.theta = [.0, .0] if theta is None else theta

    def estimate(self, mileage: int, model: str = '') -> int:
        if path.isfile(model):
            with open(model, 'r') as f:
                self.theta = list(map(float, f.readline().split(',')))
        return int(self.theta[0] + self.theta[1] * mileage)

    def predict(self, x: np.array) -> np.array:
        y_pred = np.array([self.estimate(i) for i in x])
        return y_pred

    def update_theta(self, x: np.array, y: np.array) -> None:
        y_pred = self.predict(x)
        m = len(y)
        self.theta[0] -= (self.lr * ((1 / m) * sum(y_pred - y)))
        self.theta[1] -= (self.lr * ((1 / m) * sum((y_pred - y) * x)))

    @staticmethod
    def mean_squared_error(y_pred: np.array, target: np.array) -> float:
        return np.average((target - y_pred) ** 2)

    def fit(self, data: np.array, target: np.array, iterations: int = 1000) -> tuple:

        self.theta[0] = 0.01
        self.theta[1] = 0.01

        previous_mse = None
        mse = []
        weights = []

        for i in range(iterations):
            y_pred = self.predict(data)
            current_mse = self.mean_squared_error(y_pred, target)
            if previous_mse and abs(previous_mse - current_mse) <= self.eps:
                break
            previous_mse = current_mse
            mse.append(current_mse)
            weights.append(self.theta[1])

            self.update_theta(data, target)

            print(f"Epoch [{i + 1}]:\tMSE {current_mse:.5}\t theta_1 {self.theta[1]:.4}\ttheta_0 {self.theta[0]:.4}")

        return self.theta[1], self.theta[0]

    def save(self, filename: str = 'model.conf') -> None:
        with open(filename, 'w') as f:
            f.write(','.join(map(str, self.theta)))
