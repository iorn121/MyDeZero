from mydezero import utils
import numpy as np
from mydezero.core import Function,as_variable

class Sin(Function):
    def forward(self,x):
        y=np.sin(x)
        return y
    
    def backward(self, gy):
        x,=self.inputs
        gx=gy*cos(x)
        return gx


def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y=np.cos(x)
        return y
    
    def backward(self, gy):
        x,=self.inputs
        gx=-gy*sin(x)
        return gx
    
def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y=np.tanh(x)
        return y
    
    def backward(self, gy):
        y=self.outputs[0]()
        gx=gy*(1-y*y)
        return gx
    
def tanh(x):
    return Tanh()(x)

class Reshape(Function):
    def __init__(self,shape):
        self.shape = shape
        
    def forward(self,x):
        self.x_shape = x.shape
        y=x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy,self.x_shape)


def reshape(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        y=np.transpose(x)
        return y

    def backward(self, gy):
        gx=transpose(gy)
        return gx
    
def transpose(x):
    return Transpose()(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        
        
    def forward(self, x):
        self.x_shape = x.shape
        y=x.sum(axis=self.axis,keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy=utils.reshape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
        gx=broadcast_to(gy,self.x_shape)
        return gx
    
def sum(x,axis=None,keepdims=False):
    return Sum(axis,keepdims)(x)


class BroadcastTo(Function):
    def __init__(self,shape):
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y=np.broadcast_to(x,self.shape)
        return y
    
    def backward(self, gy):
        gx=sum_to(gy,self.x_shape)
        return gx
    
def broadcast_to(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self,shape):
        self.shape = shape
        
    def forward(self,x):
        self.x_shape = x.shape
        y=utils.sum_to(x,self.shape)
        return y
    
    def backward(self,gy):
        gx=broadcast_to(gy,self.x_shape)
        return gx
    
def sum_to(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x,W):
        y=x.dot(W)
        return y
    
    def backward(self, gy):
        x,W=self.inputs
        gx=matMul(gy,W.T)
        gW=matMul(x.T,gy)
        return gx,gW

def matMul(x,W):
    return MatMul()(x,W)

class MeanSquaredError(Function):
    def forward(self,x0,x1):
        diff=x0-x1
        y=(diff**2).sum()/len(diff)
        return y
    
    def backward(self,gy):
        x0,x1=self.inputs
        diff=x0-x1
        gy=broadcast_to(gy,diff.shape)
        gx0=gy*diff*(2./len(diff))
        gx1=-gx0
        return gx0,gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)
    
def linear_simple(x,W,b=None):
    x,W=as_variable(x),as_variable(W)
    t=matMul(x,W)
    if b is None:
        return t
    
    y=t+b
    t.data=None
    return y

def sigmoid_simple(x):
    x=as_variable(x)
    y=1/(1+exp(-x))
    return y
class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matMul(gy, W.T)
        gW = matMul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)