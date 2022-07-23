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
            # x,y=f.input,f.output
            # x.grad=f.backward(y.grad)
            gys=[output.grad for output in f.outputs]
            gxs=f.backward(*gys)
            
            if not isinstance(gxs, tuple):
                gxs=(gxs,)
            for x,gx in zip(f.inputs,gxs):
                x.grad=gx
                if x.creater is not None:
                    funcs.append(x.creater)

class Function:
    """Treat every function as a class that extends Function class
    """
    def __call__(self,*inputs):
        # データを取り出す
        xs=[x.data for x in inputs]
        
        # 計算内容
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]
        
        # make Variable remember Function as parent
        for output in outputs:
            output.set_creater(self)
        self.inputs=inputs
        self.outputs=outputs
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self,xs):
        # メソッドは継承して実装
        raise NotImplementedError()

    def backward(self,gys):
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
        x=self.inputs[0].data
        gx=2*x*gy
        return gx

def square(x):
    return Square()(x)

class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
    def backward(self,gy):
        x=self.inputs[0].data
        gx=np.exp(x)*gy
        return gx

def exp(x):
    return Exp()(x)

class Add(Function):
    def forward(self,x0,x1):
        y=x0 + x1
        return (y,)
    
    def backward(self,gy):
        return gy,gy
    
def add(x0,x1):
    return Add()(x0,x1)
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

    a=Variable(np.array(2.0))
    b=Variable(np.array(3.0))
    c=add(square(a),square(b))
    c.backward()
    print(c.data)
    print(a.grad)
    print(b.grad)


if __name__ == "__main__":
    main()