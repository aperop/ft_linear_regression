import numpy as np
import matplotlib.pyplot as plt
import sys

class LinReg:
    def __init__(self, learning_rate: float = 1, eps: float = 1e-6) -> None:
        self.lr = learning_rate
        self.eps = eps
        self.theta = [.0, .0]

    def estimate(self, mileage) -> float:
        return self.theta[0] + self.theta[1] * mileage

    def predict(self, x: np.array) -> np.array:
        y_pred = np.array([self.estimate(i) for i in x])
        return y_pred

    def update_theta(self, x, y, lr):
        y_pred = self.predict(x)
        m = len(y)
        self.theta[0] -= (lr * ((1 / m) * sum(y_pred - y)))
        self.theta[1] -= (lr * ((1 / m) * sum((y_pred - y) * x)))

    @staticmethod
    def mean_squared_error(y_pred: np.array, target: np.array) -> float:
        return np.average((target - y_pred) ** 2)

    def fit(self, data, target, iterations: int = 1000) -> tuple:

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

            self.update_theta(data, target, self.lr)

            print(f"Epoch [{i + 1}]:\tMSE {current_mse:.5}\t theta_1 {self.theta[1]:.4}\ttheta_0 {self.theta[0]:.4}")

        return self.theta[1], self.theta[0]


def test():

    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
                  55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
                  45.41973014, 54.35163488, 44.1640495, 58.16847072, 56.72720806,
                  48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    Y = np.array([31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513,
                  78.21151827, 79.64197305, 59.17148932, 75.3312423, 71.30087989,
                  55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
                  60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
    X = X / np.max(X)
    Y = Y / np.max(Y)
    print(LinReg().fit(X, Y))



def main() -> None:
    if len(sys.argv) == 2:
        try:
            mileage = int(sys.argv[1])
            if mileage < 0:
                raise ValueError
            test()
            # data = np.genfromtxt('data.csv', skip_header=1, delimiter=',')
            # X = data[:, 0] / np.max(data[:, 0])
            # Y = data[:, 1] / np.max(data[:, 1])
            # print(LinReg().fit(X, Y))
            # print(f'Estimate price is {int(LinReg().estimate(mileage))} for {mileage} km')
        except ValueError:
            print('Mileage must be positive integer')
    else:
        print('Usage: python predict.py mileage')


if __name__ == '__main__':
    main()
