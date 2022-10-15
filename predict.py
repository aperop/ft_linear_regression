import sys


def predict(x: int, theta0: int = 0, theta1: int = 0) -> int:
    return theta0 + (theta1 * x)


def main() -> None:
    if len(sys.argv) == 2:
        try:
            mileage = int(sys.argv[1])
            if mileage < 0:
                raise ValueError
            print(f'Estimate price is {predict(mileage)} for {mileage} km')
        except ValueError:
            print('Mileage must be positive integer')
    else:
        print('Usage: python predict.py mileage')


if __name__ == '__main__':
    main()
