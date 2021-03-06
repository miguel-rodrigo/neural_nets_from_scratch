import numpy as np
from layer import Layer
import activations


class Model:
    # TODO:
    #   - Gradient checking: I believe my gradients are off...results are awful :(
    #   - Make the parameters update function depend on the optimizer
    #   - Make predictions on validation set during training
    #   - Create plot of training loss and validation loss during training process
    #   - Add regularization options

    def __init__(self, n_units_vector, activations, parameters=None):
        self.input_size = n_units_vector[0]
        self.layer_sizes = n_units_vector[1:]

        # Combine n_units of the next with n_units of the prev layer:
        dim_vector = zip(n_units_vector[:-1], n_units_vector[1:])
        layer_info = zip(dim_vector, activations)

        self.layers = [Layer(units_prev, units_next, activation) for
                       (units_prev, units_next), activation in layer_info]

        if parameters is not None:
            # TODO: is it better to use try/except instead? research that
            # Make sure parameters has the correct length
            assert len(parameters) == len(self.layers), \
                "Length of parameters list ({}) does not match #of layers {}" \
                .format(len(parameters), len(self.layers))
            # Make sure parameters is a list of dicts
            assert isinstance(parameters, list)
            assert isinstance(parameters[0], dict)

            for layer, params in zip(self.layers, parameters):
                layer.W = params['W']
                layer.b = params['b']

        # TODO: Make loss function modular
        def loss_function(Y, Y_hat):
            m = Y.shape[1]
            return 1 / m * np.sum(-Y * np.log(Y_hat) - (1 - Y) * np.log(1 - Y_hat))

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
            s += "Layer " + str(i + 1) + ":\n"
            s += "----------\n"
            s += "# of units: " + str(self.layer_sizes[i]) + "\n"
            s += "Activation: " + str(layer.activation_class) + "\n"
            s += "Parameter dimensions: " + "\n"
            s += "  - W: " + str(layer.W.shape) + "\n"
            s += "  - b: " + str(layer.b.shape) + "\n"

        return s

    @property
    def parameters(self):
        parameters = [{'W': layer.W, 'b': layer.b} for layer in self.layers]
        return parameters

    @property
    def gradients(self):
        parameters = [{'dW': layer.dW, 'db': layer.db} for layer in self.layers]
        return parameters

    # TODO: This is a pass through the whole training set. Add minibatch option!
    def train(self, X, Y, n_epochs=100, learning_rate=0.003):
        for i in range(n_epochs):
            # 1. Forward pass
            A_prev = X
            for layer in self.layers:
                A_prev = layer.forward_pass(A_prev)

            # 2. Compute and return loss for evaluation
            Y_hat = A_prev  # Previous activation of the last layer is actually the prediction
            loss = self.loss_function(Y, Y_hat)
            yield loss

            # 3. Backward pass
            dA = np.divide(-Y, Y_hat) + np.divide(1 - Y, 1 - Y_hat)
            for i in range(len(self.layers)-1, 0, -1):
                dA = self.layers[i].backward_pass(prev_A=self.layers[i-1].cache['A'], prev_dA=dA)

            # The previous activation for the first layer is simple X
            _ = self.layers[0].backward_pass(prev_A=X, prev_dA=dA)

            # 4. Update parameters
            self.update_parameters(learning_rate=learning_rate)

    def update_parameters(self, learning_rate=0.05):
        for layer in self.layers:
            layer.W = layer.W - learning_rate * layer.dW
            layer.b = layer.b - learning_rate * layer.db

    def predict(self, X, return_probabilities=False):
        A_prev = X
        for layer in self.layers:
            A_prev = layer.forward_pass(A_prev)

        # The last "previous" activation is actually the prediction
        if return_probabilities:
            y_hat = A_prev
        else:
            y_hat = np.float_(A_prev >= 0.5)

        return y_hat


if __name__ == "__main__":
    import utils

    np.random.seed(1)

    n_units = [2, 4, 2, 1]
    activations = [activations.ReLU, activations.ReLU, activations.Sigmoid]
    model = Model(n_units, activations)

    X, Y = utils.create_fake_data(
        random_jitter_strength=0,
        wrong_label_ratio=0
    )
    n_epochs = 1000000
    for i, iteration_loss in enumerate(model.train(X, Y, n_epochs=n_epochs, learning_rate=0.008)):
        if i % 10000 == 0 or i == n_epochs - 1:
            print("Loss on iteration {}: {}".format(i, iteration_loss))

    utils.draw_decision_boundary(model, X, Y)

    diffs = utils.gradient_checking(model, X, Y)
