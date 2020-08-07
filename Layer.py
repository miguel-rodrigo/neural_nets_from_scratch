import numpy as np


class Layer:
    def __init__(self, activation, n_units=0, n_units_next=0):
        self.W = np.random.standard_normal((n_units_next, n_units))
        self.b = np.zeros((n_units, 1))

        self.dW = np.array(())
        self.db = np.array(())

        # Cache:
        self.cache = {
            'Z': np.array([]),
            'A': np.array([])
        }

        self.activation = activation.activation
        self.activation_derivative = activation.derivative

    def forward_pass(self, X):
        Z = np.dot(self.W, X) + self.b
        self.cache['Z'] = Z
        return self.activation(Z)

    def backward_pass(self, prev_cache):
        m = self.cache['Z'].shape[1]

        dZ = np.multiply(prev_cache['dA'], self.activation_derivative(self.cache['Z']))
        self.dW = 1/m * np.dot(dZ, prev_cache['A'].T)
        self.db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(self.W.T, dZ)

        return dA
