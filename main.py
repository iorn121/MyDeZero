import math
import numpy as np
from mydezero import Variable, Function
from mydezero.utils import _dot_var, _dot_func, plot_dot_graph
import matplotlib.pyplot as plt
import mydezero.functions as F

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
    def predict(x):
        y=F.matMul(x,W)+b
        return y
    
    np.random.seed(0)
    x=np.random.rand(100,1)
    y=5+2*x+np.random.rand(100,1)
    x,y=Variable(x),Variable(y)
    # plt.scatter(x,y)
    # plt.show()
    W=Variable(np.zeros((1,1)))
    b=Variable(np.zeros(1))
    
    def mean_squared_error(x0,x1):
        diff=x0-x1
        return F.sum(diff**2)/len(diff)
    
    lr=0.1
    iters=100
    for i in range(iters):
        y_pred=predict(x)
        loss=mean_squared_error(y,y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()
        
        print(W,b,loss)
        W.data-=lr*W.grad.data
        b.data-=lr*b.grad.data

if __name__ == "__main__":
    main()
