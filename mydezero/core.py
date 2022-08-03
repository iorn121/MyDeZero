import math
import numpy as np
import contextlib
import weakref


class Config:
    enable_back_prop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_back_prop", False)


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
    """
    Treat every number as Variable class
    Each Variable has its own gradient
    """
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

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
        p = str(self.data).replace("\n", "\n"+" "*9)
        return f"variable({p})"

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation+1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # self.grad =np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))

        # implement by recursion
        # f=self.creator
        # if f is not None:
        #     x=f.input
        #     x.grad=f.backward(self.grad)
        #     x.backward()

        #  implement by loop
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            # x,y=f.input,f.output
            # x.grad=f.backward(y.grad)
            gys = [output().grad for output in f.outputs]
            with using_config('enable_back_prop', create_graph):

                gxs = f.backward(*gys)

                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad += gx
                    if x.creator is not None:
                        add_func(x.creator)

                # reset grad of variables used along the way
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    """Treat every function as a class that extends Function class
    """

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        # データを取り出す
        xs = [x.data for x in inputs]

        # 計算内容
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_back_prop:
            self.generation = max([x.generation for x in inputs])
            # make Variable remember Function as parent
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        # メソッドは継承して実装
        raise NotImplementedError()

    def backward(self, gys):
        # メソッドは継承して実装
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x)*gy
        return gx


def exp(x):
    return Exp()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data-eps)

    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data-y0.data)/(2*eps)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0*x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy*x1, gy*x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0-x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = (x**(c-1))*c*gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
