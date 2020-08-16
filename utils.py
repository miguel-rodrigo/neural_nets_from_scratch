import numpy as np
import matplotlib.pyplot as plt


def create_fake_data(num_points=200, num_petals=8, random_jitter_strength=0.02,
                     wrong_label_ratio=0.1):
    """
    Creates mock data in order to test code. The data created is a 2D flower pattern, where each petal belongs to a
    certain class depending on the octant where it lies.

    It is possible to make data noisy by adding random jitter to the position of the points so that they don't like
    precisely on the petal contour. Additionally, it is possible to create a random set of incorrect labels. This noise
    makes it possible to test models robustness.
    :param num_points: number of examples the dataset will have.
    :param num_petals: number of petals the flower will have, therefore what shape will the data have in the 2D space.
    :param random_jitter_strength: how much random noise will be added to the horizontal and vertical components of the
    data.
    :param wrong_label_ratio: what ratio of mislabeled data will the dataset contain.
    :return: two matrices of shapes (2, num_points) and (1, num_points), containing the input data and labels
    respectively.
    """
    assert num_petals % 2 == 0, "# of petals has to be an even number"

    np.random.seed(123)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    r = np.cos(num_petals//2*theta)

    x1 = r * np.cos(theta) + np.random.standard_normal(theta.shape)*random_jitter_strength
    x2 = r * np.sin(theta) + np.random.standard_normal(theta.shape)*random_jitter_strength

    X = np.vstack((x1, x2))
    assert X.shape == (2, num_points)

    # Y: colors -> 0: blue; 1: red
    Y = np.zeros((1, num_points), dtype=np.bool_)
    for i, (xi, yi) in enumerate(zip(x1, x2)):
        if xi * yi > 0:
            if yi > xi:
                Y[:, i] = 1
            else:
                Y[:, i] = 0
        else:
            if yi > -xi:
                Y[:, i] = 1
            else:
                Y[:, i] = 0

    # Add extra noise by inverting some labels at random
    rand_idx = np.random.randint(num_points,
                                 size=int(num_points * wrong_label_ratio))

    Y[:, rand_idx] = ~Y[:, rand_idx]
    Y = Y.astype(np.float)

    return X, Y


def create_train_test_split(train_to_test_ratio=0.8):
    pass


# TODO: Study what this function does to understand/fix the output
def draw_decision_boundary(model, X, y):
    """
    Draws decision boundary of the given model. Additionally, it draws the dataset that it was trained on to help
    evaluate if the model extrapolates properly or if its overfitting.
    :param model: model object to be given as defined on the Model class
    :param X: data used for training.
    :param y: labels corresponding to that data.
    :return: None
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


def gradient_checking(model, X, Y, epsilon=1e-7):
    """
    Checks the relative difference between the gradients estimated through backprop vs. gradients estimated
    using the definition of derivatives (computing the rate of growth over a small step epsilon).

    The goal of this function is to detect errors on the backpropagation formulae.
    :param model: trained model to be tested
    :param X: data used to trained the model
    :param Y: labels used to trained the model, necesary to compute the cost function
    :param epsilon: size of the step to be used in the gradient approximations
    :return: relative difference of backprop vs. approximate gradients, by layer and set of parameters
    """
    from Model import Model

    n_units_vector = [model.input_size] + model.layer_sizes
    activations = [l.activation_class for l in model.layers]

    grads = model.gradients
    differences = [{'W': 0., 'b': 0.}] * len(grads)

    for i in range(len(grads)):
        # Once for W...
        theta_plus = model.parameters
        theta_plus[i]['W'] = theta_plus[i]['W'] + epsilon
        tmp_model = Model(n_units_vector, activations, parameters=theta_plus)
        tmp_predictions = tmp_model.predict(X)
        cost_plus = model.loss_function(Y, tmp_predictions)

        theta_minus = model.parameters
        theta_plus[i]['W'] = theta_plus[i]['W'] - epsilon
        tmp_model = Model(n_units_vector, activations, parameters=theta_minus)
        tmp_predictions = tmp_model.predict(X)
        cost_minus = model.loss_function(Y, tmp_predictions)

        grad_approx_W = (cost_plus - cost_minus) / (2*epsilon)
        numerator = np.linalg.norm(grads[i]['W'] - grad_approx_W)
        denominator = np.linalg.norm(grads[i]['W']) + np.linalg.norm(grad_approx_W)
        differences[i]['W'] = numerator / denominator

        # ...and another time for b
        theta_plus = model.parameters
        theta_plus[i]['b'] = theta_plus[i]['b'] + epsilon
        tmp_model = Model(n_units_vector, activations, parameters=theta_plus)
        tmp_predictions = tmp_model.predict(X)
        cost_plus = model.loss_function(Y, tmp_predictions)

        theta_minus = model.parameters
        theta_plus[i]['b'] = theta_plus[i]['b'] - epsilon
        tmp_model = Model(n_units_vector, activations, parameters=theta_minus)
        tmp_predictions = tmp_model.predict(X)
        cost_minus = model.loss_function(Y, tmp_predictions)

        grad_approx_b = (cost_plus - cost_minus) / (2 * epsilon)
        numerator = np.linalg.norm(grads[i]['b'] - grad_approx_b)
        denominator = np.linalg.norm(grads[i]['b']) + np.linalg.norm(grad_approx_b)
        differences[i]['b'] = numerator / denominator

    return differences
