import numpy as np
import matplotlib.pyplot as plt


def create_fake_data(num_points=200, num_petals=8, random_jitter_strength=0.02,
                     wrong_label_ratio=0.1):
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
        grads = [{'W': l.W, 'b': l.b} for l in model.layers]
        grad_approx = [{'W': 0., 'b': 0.}] * len(grads)

        for i in range(len(grads)):
            theta_plus = grads
            theta_plus[i]['W'] = theta_plus[i]['W'] + epsilon

