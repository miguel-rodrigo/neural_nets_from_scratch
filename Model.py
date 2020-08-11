import numpy as np
from Layer import Layer
import Activations


class Model:
    def __init__(self, n_units_vector, activations):
        self.input_size = n_units_vector[0]
        self.layer_sizes = n_units_vector[1:]

        # Combine n_units of the next with n_units of the prev layer:
        dim_vector = zip(n_units_vector[:-1], n_units_vector[1:])
        layer_info = zip(dim_vector, activations)

        self.layers = [Layer(units_prev, units_next, activation) for
                       (units_prev, units_next), activation in layer_info]

        # TODO: Make loss function modular
        def loss_function(m, Y, Y_hat):
            return 1/m * np.sum(-Y*np.log(Y_hat) - (1-Y)*np.log(1-Y_hat))

        self.loss_function = loss_function

    def __str__(self):
        s = ""

        # General summary:
        s += "Input dimension: " + str(self.input_size) + "\n"
        s += "Output dimension: " + str(self.layer_sizes[-1]) + "\n"

        # Layer by layer:
        for i, layer in enumerate(self.layers):
            s += "\n"
            s += "----------\n"
            s += "Layer " + str(i+1) + ":\n"
            s += "----------\n"
            s += "# of units: " + str(self.layer_sizes[i]) + "\n"
            s += "Activation: " + str(layer.activation_class) + "\n"
            s += "  - W: " + str(layer.W.shape) + "\n"
            s += "  - b: " + str(layer.b.shape) + "\n"

        return s

    def train_step(self, X, Y, n_epochs, learning_rate=0.003):
        A_prev = X
        loss = np.inf
        m = X.shape[1]

        for i in range(n_epochs):
            # Forward pass
            for layer in self.layers:
                A_prev = layer.forward_pass(A_prev)

            Y_hat = A_prev
            loss = self.loss_function(m, Y, Y_hat)
            yield loss

            # Backward pass
            all_As = [layer.cache['A'] for layer in self.layers]
            # TODO: test np.multiply(-y, 1/y_hat)... for multivariate y and y_hat
            #   --> for multivariate won't work, what do we use for softmax loss?
            dA = np.multiply(-Y, 1 / Y_hat) + np.multiply((1 - Y) / (1 - Y_hat))
            # TODO: Recorrer all_As y all layers obviando la última capa
            for layer, prev_cache in zip(self.layers[::-2], all_As[1::-1]):
                dA = layer.backward_pass()


if __name__ == "__main__":
    n_units = [4, 4, 2, 1]
    activations = [Activations.ReLU, Activations.ReLU, Activations.Sigmoid]
    model = Model(n_units, activations)
