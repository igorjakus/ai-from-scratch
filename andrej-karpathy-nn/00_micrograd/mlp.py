import random
from engine import Value


class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((xi*wi for xi, wi in zip(x, self.w)), start=self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    

class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs if len(outs) != 1 else outs[0]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    

class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        inouts = [nin] + nouts
        self.layers = [Layer(inouts[i], inouts[i+1]) for i in range(len(inouts)-1)]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def gradient_descent(mlp: MLP, xs: list, ys: list, epochs = 100, lr = 0.1):
    for _ in range(epochs):
        # forward pass
        ypred = [mlp(x) for x in xs]
        loss = sum((yp - y)**2 for yp, y in zip(ypred, ys))

        # backward pass
        for p in mlp.parameters():
            p.grad = 0
        loss.backward()

        # update
        for p in mlp.parameters():
            p.data -= lr * p.grad
