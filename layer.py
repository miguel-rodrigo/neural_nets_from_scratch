import numpy as np


class Layer:
    def __init__(self, n_units, n_units_next, activation_class):
        self.W = np.random.standard_normal((n_units_next, n_units)) * 0.01
        self.b = np.zeros((n_units_next, 1))

        self.dW = np.array(())
        self.db = np.array(())

        # Cache:
        self.cache = {
            'Z': np.array([]),
            'A': np.array([])
        }

        self.activation_class = activation_class

    def forward_pass(self, X):
        Z = np.dot(self.W, X) + self.b
        self.cache['Z'] = Z

        A = self.activation_class.forward(Z)
        self.cache['A'] = A

        return A

    def backward_pass(self, prev_A, prev_dA):
        m = self.cache['Z'].shape[1]

        dZ = np.multiply(prev_dA, self.activation_class.backward(self.cache['Z']))
        self.dW = 1/m * np.dot(dZ, prev_A.T)
        self.db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(self.W.T, dZ)

        return dA
