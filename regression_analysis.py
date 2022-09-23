import matplotlib.pyplot as plt
import numpy as np

DATA_NUM = 50
DATA_RANGE = 8
NOISE_RATE = 0.1

PARAM_NUM = 6

def calc_a(x, param_num):
    a = np.empty((param_num, param_num))

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i][j] = np.sum(x ** (i+j))

    return a

def calc_b(x, y, param_num):
    b = np.empty(param_num)

    for i in range(b.shape[0]):
        b[i] = np.sum((x**i) * y)

    return b

def main():
    train_x = np.random.rand(DATA_NUM)*DATA_RANGE - (DATA_RANGE/2)

    noise = np.random.randn(DATA_NUM) * NOISE_RATE
    y = np.sin(train_x) + noise

    a = calc_a(train_x, PARAM_NUM)
    b = calc_b(train_x, y, PARAM_NUM)
    w = np.dot(np.linalg.inv(a), b.reshape(-1, 1))

    f = np.poly1d(w.flatten()[::-1])

    test_x = np.linspace(-(DATA_RANGE / 2), (DATA_RANGE / 2), 100)

    plt.title("Result")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

    # Data
    plt.scatter(train_x, y, label="Data")

    # Regression
    plt.plot(test_x, f(test_x), color="green", label="Regression")

    # Golden
    plt.plot(test_x, np.sin(test_x), color="blue", label="Golden")

    plt.legend(loc=0)
    plt.show()

if __name__ == "__main__":
    main()