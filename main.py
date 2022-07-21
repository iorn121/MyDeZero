import numpy as np
class Variable:
    """
    Treat every number as Variable class
    Each Variable has its own gradient
    """
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.grad=None
        self.creater=None
    
    def set_creater(self,func):
        self.creater = func
        
    def backward(self):
        if self.grad is None:
            self.grad =np.ones_like(self.data)
        
        # implement by recursion
        # f=self.creater
        # if f is not None:
        #     x=f.input
        #     x.grad=f.backward(self.grad)
        #     x.backward()
        
        #  implement by loop
        funcs=[self.creater]
        while funcs:
            f=funcs.pop()
            x,y=f.input,f.output
            x.grad=f.backward(y.grad)

            if x.creater is not None:
                funcs.append(x.creater)

class Function:
    """Treat every function as a class that extends Function class
    """
    def __call__(self,input: Variable):
        # データを取り出す
        x=input.data
        
        # 計算内容
        y=self.forward(x)
        output=Variable(as_array(y))
        
        # make Variable remember Function as parent
        output.set_creater(self)
        self.input=input
        self.output=output
        return output

    def forward(self,x):
        # メソッドは継承して実装
        raise NotImplementedError()

    def backward(self,gy):
        # メソッドは継承して実装
        raise NotImplementedError()

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Square(Function):
    def forward(self,x):
        return x**2
    
    def backward(self,gy):
        x=self.input.data
        gx=2*x*gy
        return gx

def square(x):
    return Square()(x)

class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
    def backward(self,gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx

def exp(x):
    return Exp()(x)

def numerical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)

    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)

def main():

    x=Variable(np.array(0.5))

    print("x:",x.data)
    y=square(exp(square(x)))
    print("y:",y.data)

    # manual back propagation
    # y.grad=np.array(1.0)
    # print("y_grad:",y.grad)
    # b.grad=C.backward(y.grad)
    # a.grad=B.backward(b.grad)
    # x.grad=A.backward(a.grad)
    # print("x_grad:",x.grad)

    # check nodes of graph in the reverse direction
    # assert y.creater==C
    # assert y.creater.input==b
    # assert y.creater.input.creater==B
    # assert y.creater.input.creater.input==a
    # assert y.creater.input.creater.input.creater==A
    # assert y.creater.input.creater.input.creater.input==x

    # automatic back propagation
    y.backward()
    print("x_grad:",x.grad)


if __name__ == "__main__":
    main()