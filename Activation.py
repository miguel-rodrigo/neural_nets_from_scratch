import numpy as np


class Activation:
    @staticmethod
    def activation(x):
        raise NotImplementedError

    @staticmethod
    def derivative(x):
        raise NotImplementedError


class ReLU(Activation):
    @staticmethod
    def activation(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return np.where(x >= 0, 1, 0)


class Sigmoid(Activation):
    @staticmethod
    def activation(x):
        return 1 / (1+np.exp(-x))

    @staticmethod
    def derivative(x):
        return Sigmoid.activation(x) * (1-Sigmoid.activation(x))


class Linear(Activation):
    @staticmethod
    def activation(x):
        return x

    @staticmethod
    def derivative(x):
        return 1
