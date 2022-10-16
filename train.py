from LinReg import LinReg
import numpy as np
# from sklearn.linear_model import LinearRegression


def train() -> None:
    data = np.genfromtxt('data.csv', skip_header=1, delimiter=',')
    X = data[:, 0]
    Y = data[:, 1]
    model = LinReg()
    model.fit(X, Y)
    print(model.theta[0], model.theta[1])
    model.save()


if __name__ == '__main__':
    train()
