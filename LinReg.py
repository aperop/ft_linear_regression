import matplotlib.pyplot as plt
from typing import Any
import pandas as pd


class LinReg:

    def __init__(self, csv_data: str) -> None:

        file = pd.read_csv(csv_data)
        data = file.values.reshape(-1, 2)
        self.headers = file.columns
        self.x, self.y = data[:, 0], data[:, 1]
        try:
            with open('weight.csv', 'r') as f:
                raw = f.read()
            data = raw.split(',')
            self.theta0 = float(data[0])
            self.theta1 = float(data[1])
        except IOError:
            self.theta0, self.theta1 = 0, 0
        self.cost_history = []
        self.r2_score = None

    def fit(self, epoch: int = 100000, lr: float = 0.1, eps: float = 1e-3) -> tuple:
        self.cost_history = []
        self.theta0, self.theta1 = 0, 0

        for i in range(epoch):
            self.__update_thetas(lr)
            current_cost = self.cost()
            if self.cost_history and abs(current_cost - self.cost_history[-1]) < eps:
                break
            self.cost_history.append(current_cost)
            print(f"Iteration {i + 1}: MSE {current_cost}, theta0 {self.theta0}, theta1 {self.theta1}")
        self.__r2_score()
        return self.theta0, self.theta1

    def __r2_score(self) -> None:
        normalized_x = self.__normalize(self.x)
        u = sum([(self.__estimate(x) - y) ** 2 for x, y in zip(normalized_x, self.y)])
        v = sum([(y - self.y.mean()) ** 2 for y in self.y])
        self.r2_score = 1 - u/v

    def predict(self, x: int):
        val = self.__normalize(x)
        return self.__estimate(val)

    def __estimate(self, x: Any) -> Any:
        return self.theta0 + self.theta1 * x

    def cost(self) -> float:
        normalized_x = self.__normalize(self.x)
        return sum([(self.__estimate(x) - y) ** 2 for x, y in zip(normalized_x, self.y)]) / len(self.x)

    def plot(self) -> object:
        line_x = [self.x.min(), self.x.max()]
        line_y = [self.__estimate(x) for x in [0, 1]]
        plt.figure(figsize=(9, 7))
        plt.scatter(self.x, self.y, color='green', alpha=.5, label="data")
        plt.plot(line_x, line_y, color='red', alpha=.7, label="model")
        plt.xlabel(self.headers[0])
        plt.ylabel(self.headers[1])
        plt.title(self.__module__)
        plt.legend()
        plt.grid()
        return plt

    def plot_cost(self) -> plt.plot:
        x = range(len(self.cost_history))
        y = self.cost_history
        plt.figure(figsize=(9, 7))
        plt.plot(x, y, label="cost")
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Cost Function')
        plt.legend()
        return plt

    def score(self) -> float:
        return self.r2_score

    def save(self) -> None:
        with open('weight.csv', "w") as f:
            f.write(f'{self.theta0},{self.theta1}')

    def __normalize(self, x: Any) -> Any:
        return (x - self.x.min()) / (self.x.max() - self.x.min())

    def __update_thetas(self, lr: float) -> None:
        normalized_x = self.__normalize(self.x)
        theta0_tmp_list = []
        theta1_tmp_list = []

        for x, y in zip(normalized_x, self.y):
            theta0_tmp_list.append(self.__estimate(x) - y)
            theta1_tmp_list.append((self.__estimate(x) - y) * x)

        x_size = len(normalized_x)
        self.theta0 = self.theta0 - lr * (sum(theta0_tmp_list) / x_size)
        self.theta1 = self.theta1 - lr * (sum(theta1_tmp_list) / x_size)
