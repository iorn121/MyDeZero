import math
import numpy as np
from mydezero import Variable, Function
from mydezero.utils import _dot_var, _dot_func, plot_dot_graph
import matplotlib.pyplot as plt
import mydezero.functions as F
import mydezero.layers  as L

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
    np.random.seed(0)
    x=np.random.rand(100,1)
    y=np.sin(2*np.pi*x)+np.random.rand(100,1)
    l1=L.Linear(10)
    l2=L.Linear(1)

    def predict(x):
        y=l1(x)
        y=F.sigmoid_simple(y)
        y=l2(y)
        return y
    
    lr=0.2
    iters=1000

    for i in range(iters):
        y_pred= predict(x)
        loss=F.mean_squared_error(y,y_pred)

        l1.cleargrad()
        l2.cleargrad()
        loss.backward()

        for l in [l1,l2]:
            for p in l.params():
                p.data-=lr*p.grad.data
        
        if i%100==0:
            print(loss)

if __name__ == "__main__":
    main()
