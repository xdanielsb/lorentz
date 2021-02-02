"""
    author: xdanielsb
    date: 2-feb-2021
"""

import decimal

import numpy as np
from scipy import stats

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

a = 10
b = 3
c = 28
d = -16
k = 0.2


def lorentz(x1, x2, x3, x4, h):
    global a, b, c, d, k
    x1_ = a * (x2 - x1)
    x2_ = c * x1 + x2 - x1 * x3 - x4
    x3_ = x1 * x2 - b * x3
    x4_ = k * x2 * x3
    return x1_ * h, x2_ * h, x3_ * h, x4_ * h


def getValuesLorentz(x1, x2, x3, x4, iters=1000, h=0.01):
    """
    Apply Runge Kutta 4 for getting the values
    """
    xs, ys, zs = [], [], []
    for i in range(1, iters + 1):
        xs.append(x1)
        ys.append(x2)
        zs.append(x3)
        a1, a2, a3, a4 = lorentz(x1, x2, x3, x4, h)
        b1, b2, b3, b4 = lorentz(x1 + a1 / 2, x2 + a2 / 2, x3 + a3 / 2, x4 + a4 / 2, h)
        c1, c2, c3, c4 = lorentz(x1 + b1 / 2, x2 + b2 / 2, x3 + b3 / 2, x4 + b4 / 2, h)
        d1, d2, d3, d4 = lorentz(x1 + c1, x2 + c2, x3 + c3, x4 + c4, h)
        x1 = x1 + (a1 + 2 * b1 + 2 * c1 + d1) / 6
        x2 = x2 + (a2 + 2 * b2 + 2 * c2 + d2) / 6
        x3 = x3 + (a3 + 2 * b3 + 2 * c3 + d3) / 6
        x4 = x4 + (a4 + 2 * b4 + 2 * c4 + d4) / 6
    return xs, ys, zs


if __name__ == "__main__":

    iters = 5000
    xs, ys, zs = getValuesLorentz(10, 10, 10, -10, iters)

    # Plot the surface.
    fig = plt.figure(1)
    ax = fig.gca(projection="3d")
    surf1 = ax.scatter(np.array(xs), np.array(ys), np.array(zs), c="r", label="iter 1")
    xs2, ys2, zs2 = getValuesLorentz(0, 0, 100, -2, iters)
    xs3, ys3, zs3 = getValuesLorentz(2, 10, 100, -2, iters)
    surf2 = ax.scatter(
        np.array(xs2), np.array(ys2), np.array(zs2), c="b", label="iter 2"
    )
    fig = plt.figure(2)

    xx = [x for x in range(len(xs))]

    # plot the variation of xi 
    plt.plot(xx, xs, label="iter1")
    plt.plot(xx, xs2, label="iter2")

    # compute lyapunov coefficient
    diff = np.power(np.abs(np.array(xs) - np.array(xs2)), 2)
    diff += np.power(np.abs(np.array(ys) - np.array(ys2)), 2)
    diff += np.power(np.abs(np.array(zs) - np.array(zs2)), 2)
    diff = np.log(diff)

    slope, intercept, r_value, p_value, std_err = stats.linregress(xx, diff)

    # omg the tricky part :'( :0
    print(
        "slope={}, \nintercept={},\n r_value={},\n p_value={},\n std_err={})".format(
            slope, intercept, r_value, p_value, std_err
        )
    )

    plt.plot(xx, diff, label="diff")
    plt.plot(np.array(xx), np.array(xx) * float(slope) + float(intercept))
    plt.legend()
    plt.show()
