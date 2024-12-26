from dezero.core import Variable, add
import numpy as np
from dezero.functions import square

def main():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()
    print(y.data)
    print(x.grad)

if __name__ == "__main__":
    main()