import numpy as np
import contextlib
import weakref
class Config:
    enable_back_prop=True
    

@contextlib.contextmanager
def using_config(name,value):
    old_value=getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)

def no_grad():
    return using_config("enable_back_prop",False)
class Variable:
    """
    Treat every number as Variable class
    Each Variable has its own gradient
    """
    def __init__(self,data,name=None):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.name=name
        self.grad=None
        self.creater=None
        self.generation=0
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p=str(self.data).replace("\n","\n"+" "*9)
        return f"variable({p})"
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __add__(self, other):
        return add(self, other)
    
    def set_creater(self,func):
        self.creater = func
        self.generation=func.generation+1
        
    def backward(self,retain_grad=False):
        if self.grad is None:
            self.grad =np.ones_like(self.data)
        
        # implement by recursion
        # f=self.creater
        # if f is not None:
        #     x=f.input
        #     x.grad=f.backward(self.grad)
        #     x.backward()
        
        #  implement by loop
        funcs=[]
        seen_set=set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creater)
        while funcs:
            f=funcs.pop()
            # x,y=f.input,f.output
            # x.grad=f.backward(y.grad)
            gys=[output().grad for output in f.outputs]
            gxs=f.backward(*gys)
            
            if not isinstance(gxs, tuple):
                gxs=(gxs,)
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad+=gx
                if x.creater is not None:
                    add_func(x.creater)
            
            # reset grad of variables used along the way
            if not retain_grad:
                for y in f.outputs:
                    y().grad=None
    
    def cleargrad(self):
        self.grad=None

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
        if Config.enable_back_prop:
            self.generation=max([x.generation for x in inputs])
            # make Variable remember Function as parent
            for output in outputs:
                output.set_creater(self)
            self.inputs=inputs
            self.outputs=[weakref.ref(output) for output in outputs]
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

class Mul(Function):
    def forward(self,x0,x1):
        y=x0*x1
        return y
    
    def backward(self,gy):
        x0,x1=self.inputs[0].data,self.inputs[1].data
        return gy*x1,gy*x0

def mul(x0,x1):
    return Mul()(x0,x1)

def main():

    x=Variable(np.array(0.5))

    print("x:",x)
    y=square(exp(square(x)))
    print("y:",y)

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
    print(f"back propagation: {Config.enable_back_prop}")
    print("x_grad:",x.grad)

    with no_grad():
        a=Variable(np.array(2.0))
        b=Variable(np.array(3.0))
        c=square(a)*square(b)+a+b
        print(c)


if __name__ == "__main__":
    main()