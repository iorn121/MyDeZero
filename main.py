import math
import numpy as np
from mydezero import Variable,Function
from mydezero.utils import _dot_var,_dot_func,plot_dot_graph

def sphere(x,y):
    z=x**2+y**2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


class Sin(Function):
    def forward(self,x):
        y=np.sin(x)
        return y
    
    def backward(self,gy):
        x=self.inputs[0].data
        gx=np.cos(x)*gy
        return gx

def sin(x):
    return Sin()(x)


def my_sin(x,threshold=1e-4):
    y=0
    for i in range(100000):
        c=(-1)**i/math.factorial(2*i+1)
        t=c*x**(2*i+1)
        y+=t
        if abs(t.data)<threshold:
            break
    return y

def main():

    x0=Variable(np.array(np.pi/4))
    x1=Variable(np.array(np.pi/4))
    y=sin(x0)
    z=my_sin(x1)
    y.backward()
    z.backward()
    print("y:",y,"y.grad:",x0.grad)
    print("z:",z,"z.grad:",x1.grad)
    y.name="y"
    z.name="z"
    plot_dot_graph(y,verbose=False,to_file="sin_graph.png")
    plot_dot_graph(z,verbose=False,to_file="my_sin_graph.png")


if __name__ == "__main__":
    main()