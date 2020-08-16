import numpy as np

# TODO: create additional decorator for a try/catch block that checks dimensions of x


class Activation:
    @staticmethod
    def forward(x):
        raise NotImplementedError

    @staticmethod
    def backward(x):
        raise NotImplementedError


class ReLU(Activation):
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(x):
        return np.where(x >= 0, 1, 0)


class Sigmoid(Activation):
    @staticmethod
    def forward(x):
        return 1 / (1+np.exp(-x))

    @staticmethod
    def backward(x):
        return Sigmoid.forward(x) * (1-Sigmoid.forward(x))


class Linear(Activation):
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def backward(x):
        return 1
