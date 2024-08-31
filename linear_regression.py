import numpy as np
import os
import matplotlib.pyplot as plt

ORDER = 3


def synthetic_data(s, e, n, noisy=True):
    x = np.linspace(s, e, n)
    error = np.random.normal(0, 1, x.shape) * 0.1
    y = np.sin(2*np.pi*x)
    if noisy:
        y += error
    return x, y


class polynomial():
    def __init__(self, m):
        self.m = m+1
        self.w = np.array(np.random.normal(0, 1, self.m)) * 0.5

    def value(self, x):
        sum = 0
        for i in range(self.m):
            sum += self.w[i] * np.power(x, i)
        return sum


def loss(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    return 0.5 * np.sum([(y[i] - y_hat[i]) ** 2 for i in range(len(y))])


def y_hat(x, g):
    return np.array([g.value(i) for i in x])


def plot_linear_regression(x, y, a, b, g):
    plt.ioff()
    plt.figure()
    plt.scatter(x, y)
    plt.plot(a, b, color='r')
    plt.plot(a, y_hat(a, g), color='b')
    plt.fill_between(a, y_hat(a, g)+0.2, y_hat(a, g)-0.2, alpha=0.1, color='b')
    plt.title(f'The order of polynomial is : {g.m-1}')
    plt.legend(['original funciton', 'fitted funciton', 'data_with_noise'])
    plt.show()


class animation_regression():
    def __init__(self, x, y, a, b, g):
        self.a = a
        self.g = g
        plt.figure()
        plt.ion()
        self.ax = plt.gca()
        self.ax.scatter(x, y)
        self.ax.plot(a, b, color='r')
        self.ln, = self.ax.plot(a, y_hat(a, g), color='b')
        plt.title(f'The order of polynomial is : {g.m-1}')
        plt.legend(['original funciton', 'fitted funciton', 'data_with_noise'])

    def update_g(self, g):
        self.ln.set_ydata(y_hat(self.a, g))
        plt.draw()
        plt.pause(0.1)


def update_g_w(x, y, g, lr):
    X = np.mat([np.power(x, j) for j in range(g.m)])
    w = np.mat(g.w).reshape(-1, 1)
    y = np.mat(y).reshape(-1, 1)
    grad = np.array((X * (X.T * w - y))).flatten()
    g.w = g.w - lr * grad


def analytical_solution_w(x, y, g):
    X = np.mat([np.power(x, j) for j in range(g.m)]).T
    y = np.mat(y).reshape(-1, 1)
    g.w = np.array(np.linalg.inv(X.T * X) * X.T * y).flatten()


if __name__ == "__main__":

    os.makedirs(os.path.join('.', 'data'), exist_ok=True)
    data_file = os.path.join('.', 'data', 'linaer_regression.npy')

    # x, y = synthetic_data(0, 1, 10)
    # np.save(data_file, (x, y))

    g = polynomial(ORDER)
    x, y = np.load(data_file)
    x_original, y_original = synthetic_data(0, 1, 50, noisy=False)

    ani = animation_regression(x, y, x_original, y_original, g)

    print('The initial parameters:')
    print(g.w)
    print(f'initial loss is: {loss(y, y_hat(x,g))}')

    for i in range(200000):
        update_g_w(x, y, g, 0.01)
        if i % 1000 == 0:
            print(f'epoch is : {i} , loss is : {loss(y, y_hat(x,g))}')
            ani.update_g(g)

    # analytical_solution_w(x, y, g)

    print('The final parameters:')
    print(g.w)
    print(f'final loss is : {loss(y, y_hat(x,g))}')
    plt.close()
    # plt.show()

    plot_linear_regression(x, y, x_original, y_original, g)
