import numpy as np
from mydezero import Variable

def main():

    x=Variable(np.array(0.5))

    print("x:",x)
    y=(x+3)**2
    y.backward()
    print("y:",y)
    print("x.grad",x.grad)




if __name__ == "__main__":
    main()