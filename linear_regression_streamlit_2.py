import re
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import os
from IPython import display
from PIL import Image


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
    # plt.figure()
    plt.scatter(x, y)
    plt.plot(a, b, color='r')
    plt.plot(a, y_hat(a, g), color='b')
    plt.fill_between(a, y_hat(a, g)+0.2, y_hat(a, g)-0.2, alpha=0.1, color='b')
    plt.title(f'The order of polynomial is : {g.m-1}')
    plt.legend(['original funciton', 'fitted funciton', 'data_with_noise'])
    # plt.show()

def fig2img(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGB", (w, h), fig.canvas.tostring_rgb())
    return img

class animation_regression():
    def __init__(self, x, y, a, b, g):
        self.a = a
        self.g = g
        self.fig = plt.figure()
        # plt.ion()
        self.ax = plt.gca()
        self.ax.scatter(x, y)
        self.ax.plot(a, b, color='r')
        self.ln, = self.ax.plot(a, y_hat(a, g), color='b')
        plt.title(f'The order of polynomial is : {g.m-1}')
        plt.legend(['original funciton', 'fitted funciton', 'data_with_noise'])

    def update_g(self, g):
        self.ln.set_ydata(y_hat(self.a, g))
        # plt.draw()
        # st.pyplot(self.fig)
        # display.display(self.fig)
        # plt.pause(0.1)
        # display.clear_output(wait=True)


def update_g_w(x, y, g, lr):
    # X = np.mat([np.power(x, j) for j in range(g.m)])
    X = np.asmatrix([np.power(x, j) for j in range(g.m)])
    # w = np.mat(g.w).reshape(-1, 1)
    w = np.asmatrix(g.w).reshape(-1, 1)
    # y = np.mat(y).reshape(-1, 1)
    y = np.asmatrix(y).reshape(-1, 1)
    grad = np.array((X * (X.T * w - y))).flatten()
    g.w = g.w - lr * grad


def analytical_solution_w(x, y, g):
    # X = np.mat([np.power(x, j) for j in range(g.m)]).T
    X = np.asmatrix([np.power(x, j) for j in range(g.m)]).T
    # y = np.mat(y).reshape(-1, 1)
    y = np.asmatrix(y).reshape(-1, 1)
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

    # progress_bar =  st.sidebar.progress(0)
    # progress_text = st.sidebar.empty()


    begin_text =  st.empty()
    display_fit = st.empty()

    progress_text = st.empty()
    progress_bar =  st.progress(0)

    end_text =  st.empty()


    print('The initial parameters:')
    print(g.w)
    print(f'initial loss is: {loss(y, y_hat(x,g))}')

    # print(f'''
    #     The initial parameters:
    #     {g.w}
    #     initial loss is: {loss(y, y_hat(x,g))} ''')

    begin_text.text(f'''
        The initial parameters:
        {g.w}
        initial loss is: {loss(y, y_hat(x,g))} ''')

    # st.pyplot(ani.fig)
    for i in range(200000):
        update_g_w(x, y, g, 0.01)
        if i % 1000 == 0:
            print(f'epoch is : {i} , loss is : {loss(y, y_hat(x,g))}')
            progress_bar.progress(i/200000)
            progress_text.text(f'progress ...  {i/2000} / 100')
            ani.update_g(g)
            display_fit.pyplot(ani.fig)
            # plt.pause(0.1)
            # display.clear_output(wait=True)

    progress_bar.progress(0.999999)
    progress_text.text(f'progress ...  100 / 100')


    # analytical_solution_w(x, y, g)

    print('The final parameters:')
    print(g.w)
    print(f'final loss is : {loss(y, y_hat(x,g))}')

    end_text.text(f'''
        The final parameters:
        {g.w}
        final loss is: {loss(y, y_hat(x,g))} ''')

    # plt.close()
    # plt.show()

    # fig_2 = plt.figure()
    # plot_linear_regression(x, y, x_original, y_original, g)
    # plt.show()
    # st.pyplot(fig_2)
    # image.image(fig2img(fig_2))
    # st.image(image)
