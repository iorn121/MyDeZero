import numpy as np
from mydezero import Variable
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

def main():

    x=Variable(np.array(0.5))
    y=Variable(np.array(0.5))
    print("x:",x)
    print("y:",y)
    z=matyas(x,y)
    z.backward()
    print("z:",z)
    print("x.grad",x.grad)
    print("y.grad",y.grad)
    
    x.name="x"
    y.name="y"
    z.name="z"

    plot_dot_graph(z,verbose=False,to_file='matyas.png')



if __name__ == "__main__":
    main()