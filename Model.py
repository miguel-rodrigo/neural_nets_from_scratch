import numpy as np
from Layer import Layer
import Activations

import utils


class Model:
    # TODO:
    #   - Make the parameters update function depend on the optimizer
    #   - __call__ method that calls the predict function
    #   - Gradient checking: I believe my gradients are off...
    #   - Make predictions on validation set during training
    #   - Create plot of training loss and validation loss during training

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
            s += "Parameter dimensions: " + "\n"
            s += "  - W: " + str(layer.W.shape) + "\n"
            s += "  - b: " + str(layer.b.shape) + "\n"

        return s

    # TODO: This is a pass through the whole training set. Add minibatch option!
    def train(self, X, Y, n_epochs=100, learning_rate=0.003):
        loss = np.inf
        m = X.shape[1]

        for i in range(n_epochs):
            # 1. Forward pass
            A_prev = X
            for layer in self.layers:
                A_prev = layer.forward_pass(A_prev)

            # 2. Compute and return loss for evaluation
            Y_hat = A_prev
            loss = self.loss_function(m, Y, Y_hat)
            yield loss

            # 3. Backward pass
            # TODO: test np.multiply(-y, 1/y_hat)... for multivariate y and y_hat
            #   --> for multivariate won't work, what do we use for softmax loss?
            dA = np.multiply(-Y, 1 / Y_hat) + np.multiply((1 - Y), 1 / (1 - Y_hat))
            for layer in self.layers[::-1]:
                dA = layer.backward_pass(prev_A=layer.cache['A'], prev_dA=dA)

            # 4. Update parameters
            self.update_parameters()

    def update_parameters(self, learning_rate=0.05):
        for layer in self.layers:
            layer.W = layer.W - learning_rate * layer.dW
            layer.b = layer.b - learning_rate * layer.db

    def predict(self, X):
        A_prev = X
        for layer in self.layers:
            A_prev = layer.forward_pass(A_prev)

        # The last "previous" activation is actually the prediction
        y_hat = np.float_(A_prev >= 0.5)
        return y_hat


if __name__ == "__main__":
    n_units = [2, 2, 1]
    activations = [Activations.ReLU, Activations.Sigmoid]
    model = Model(n_units, activations)

    X, Y = utils.create_fake_data()
    n_epochs = 1000
    for i, iteration_loss in enumerate(model.train(X, Y, n_epochs=n_epochs)):
        if i % 100 == 0 or i == n_epochs-1:
            print("Loss on iteration {}: {}".format(i, iteration_loss))
