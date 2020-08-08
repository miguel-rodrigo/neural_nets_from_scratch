import numpy as np
import Layer


class Model:
    def __init__(self, n_units_vector, activations):
        # Combine n_units of the next with n_units of the prev layer:
        dim_vector = zip(n_units_vector[:-1], n_units_vector[1:])
        layer_info = zip(dim_vector, activations)

        self.layers = [Layer(activation, units_prev, units_next) for
                       (units_prev, units_next), activation in layer_info]
