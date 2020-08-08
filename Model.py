import numpy as np
from Layer import Layer
import Activation


class Model:
    def __init__(self, n_units_vector, activations):
        # Combine n_units of the next with n_units of the prev layer:
        dim_vector = zip(n_units_vector[:-1], n_units_vector[1:])
        layer_info = zip(dim_vector, activations)

        self.layers = [Layer(units_prev, units_next, activation) for
                       (units_prev, units_next), activation in layer_info]
