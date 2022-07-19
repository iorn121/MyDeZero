class Variable:
    """Treat every number as Variable class
    """
    def __init__(self,data: any):
        self.data = data

class Function:
    """Treat every function as a class that extends Function class
    """
    def __call__(self,input: Variable):
        # データを取り出す
        x=input.data
        # 計算内容
        y=self.forward(x)
        output=Variable(y)
        return output

    def forward(self,x):
        # メソッドは継承して実装
        raise NotImplementedError()


class Square(Function):
    def forward(self,x):
        return x**2

class Exp(Function):
    def forward(self,x):
        return np.exp(x)

def numerical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)


import numpy as np
x=Variable(np.array(0.5))
def f(x):
    A=Square()
    B=Exp()
    C=Square()
    return C(B(A(x)))
y=f(x)

print("x:",x.data)
dy=numerical_diff(f,x)
print("dy:",dy)
print("y:",y.data)