import numpy as np
import heapq


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func: 'Function'):
        self.creator = func
        self.generation += 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 逆伝播の呼び出す順を考慮するためにgenerationをベースにbackwardする順を制御
        funcs = []
        heapq.heapify(funcs)
        seen_set = set()

        def add_func(f: 'function'):
            if f not in seen_set:
                heapq.heappush(funcs, f)
                seen_set.add(f)

        add_func(self.creator)

        while funcs:
            f = heapq.heappop(funcs)
            gys = [output.grad for output in f.outputs]
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

    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs: Variable) -> list[Variable]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def __lt__(self, other: 'Function'):
        if self.generation is None or other.generation is None:
            raise ValueError("Generation info missed.")
        return self.generation > other.generation

    def __hash__(self):
        return id(self)

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)
