import math


class Value:

    def __init__(self, data, children=(), op=''):
        self.data = float(data)
        self.grad = 0.0

        self._prev = set(children)
        self._op = op

        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            '+'
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            '*'
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, power):

        out = Value(
            self.data ** power,
            (self,),
            f'**{power}'
        )

        def _backward():
            self.grad += (
                power *
                self.data ** (power - 1) *
                out.grad
            )

        out._backward = _backward

        return out

    def relu(self):

        out = Value(
            max(0, self.data),
            (self,),
            'ReLU'
        )

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):

        s = 1 / (1 + math.exp(-self.data))

        out = Value(
            s,
            (self,),
            'sigmoid'
        )

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward

        return out

    def sum(self):
        return self

    def backward(self):

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build(child)

                topo.append(v)

        build(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"