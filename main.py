import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from predict import predict


def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


def gradient_descent(x: np.array, y: np.array,
                     iterations: int = 1000,
                     learning_rate: float = 0.0001,
                     eps: float = 1e-6) -> tuple:

    theta1 = 0.1
    theta0 = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = (len(x))

    mse = []
    weights = []
    previous_mse = None

    for i in range(iterations):

        y_pred = predict(x, theta0, theta1)
        current_mse = mean_squared_error(y, y_pred)
        if previous_mse and abs(previous_mse - current_mse) <= eps:
            break
        previous_mse = current_mse
        mse.append(current_mse)
        weights.append(theta1)

        weight_derivative = -(2 / n) * sum(x * (y - y_pred))
        bias_derivative = -(2 / n) * sum(y - y_pred)
        theta1 = theta1 - (learning_rate * weight_derivative)
        theta0 = theta0 - (learning_rate * bias_derivative)

        print(f"Iteration {i + 1}: MSE {current_mse}, theta1 {theta1}, theta0 {theta0}")

    plt.figure(figsize=(10, 8))
    plt.plot(weights, mse, color='yellow')
    plt.scatter(weights, mse, marker='o', color='green')
    plt.title("MSE vs theta1")
    plt.ylabel("MSE")
    plt.xlabel("theta1")
    # plt.show()

    return theta1, theta0


def main():

    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
                  55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
                  45.41973014, 54.35163488, 44.1640495, 58.16847072, 56.72720806,
                  48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    Y = np.array([31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513,
                  78.21151827, 79.64197305, 59.17148932, 75.3312423, 71.30087989,
                  55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
                  60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

    estimated_weight, estimated_bias = gradient_descent(X, Y, iterations=2000)
    print(f"Estimated theta1: {estimated_weight}\nEstimated theta0: {estimated_bias}")

    Y_pred = estimated_weight * X + estimated_bias

    plt.figure(figsize=(10, 8))
    plt.scatter(X, Y, marker='o', color='red')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue', markerfacecolor='red',
             markersize=10, linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    main()
