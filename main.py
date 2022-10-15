from LinReg import LinReg
import numpy as np
import sys
from sklearn.linear_model import LinearRegression


def train() -> None:
    data = np.genfromtxt('data.csv', skip_header=1, delimiter=',')
    X = data[:, 0]
    Y = data[:, 1]
    model = LinReg()
    model.fit(X, Y)
    print(model.theta[0], model.theta[1])
    # model.save()


def predict(mileage: int) -> None:
    print(f'Estimate price is {int(LinReg().estimate(mileage, "model.conf"))} for {mileage} km')


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) == 3 and sys.argv[1] == 'predict':
        try:
            mileage = int(sys.argv[2])
            if mileage < 0:
                raise ValueError
            predict(mileage)
        except ValueError:
            print('Mileage must be positive integer')
    else:
        print('Usage: python predict.py mileage')


if __name__ == '__main__':
    main()
