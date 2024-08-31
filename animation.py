#!/usr/bin/env python
# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import psutil


def data_gen():  # 设置xy变量
    x = -1
    while True:
        y = psutil.cpu_percent(interval=1)  # 获取cpu数值,1s获取一次。
        x += 1
        yield x, y


def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,


xdata = []
ydata = []
fig = plt.figure(figsize=(18, 8), facecolor="white")
ax = fig.add_subplot(111)
line, = ax.plot(xdata, ydata, color="red")


# update the data
def update(data):
    x, y = data
    xdata.append(x)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()
    if x >= xmax:
        ax.set_xlim(0, xmax+10)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)
    return line,


ani = animation.FuncAnimation(fig, update, data_gen, blit=False, interval=1,
                              repeat=False, init_func=init)
plt.show(
