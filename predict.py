import sys
from LinReg import LinReg


def predict(mileage: int) -> None:
    price = int(LinReg().estimate(mileage, "weight.csv"))
    if price < 0:
        print('This model isn\'t suitable for this value because the mileage is too high')
    else:
        print(f'Estimate price is {price} for {mileage} km')


def main() -> None:
    if len(sys.argv) == 2:
        try:
            mileage = int(sys.argv[1])
            if mileage < 0:
                raise ValueError
            predict(mileage)
        except ValueError:
            print('Mileage must be positive integer')
    else:
        print('Usage: python predict.py mileage')


if __name__ == '__main__':
    main()
