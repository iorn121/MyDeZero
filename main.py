import math
import numpy as np
from mydezero import Variable, Function
from mydezero.utils import _dot_var, _dot_func, plot_dot_graph


def sphere(x, y):
    z = x**2+y**2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.cos(x)*gy
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=1e-4):
    y = 0
    for i in range(100000):
        c = (-1)**i/math.factorial(2*i+1)
        t = c*x**(2*i+1)
        y += t
        if abs(t.data) < threshold:
            break
    return y


def rosenbrock(x0, x1):
    y = 100*(x1-x0**2)**2+(x0-1)**2
    return y


def f(x):
    y = x**4-2*x**2
    return y


def gx2(x):
    return 12*x**2 - 4


def main():

    x = Variable(np.array(2.0))
    iters = 10
    for i in range(iters):
        print(i, x)

        y = f(x)
        x.cleargrad()
        y.backward(create_graph=True)

        gx = x.grad
        x.cleargrad()
        gx.backward()
        gx2 = x.grad

        x.data -= gx.data / gx2.data


if __name__ == "__main__":
    main()
