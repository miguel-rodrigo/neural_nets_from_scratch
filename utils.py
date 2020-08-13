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

    # Octant I: 0
    # Octants II, III: 1
    # Octant IV: 0
    # Octant V: 1
    # Octatns VI, VII: 0
    # Octact VIII: 1
    octants = np.floor(theta / (2*np.pi) * 8) + 1
    zero_octants = [1, 4, 6, 7]
    Y = [0 if octant in zero_octants else 1 for octant in octants]
    Y = np.array(Y, ndmin=2, dtype=np.bool_) # ndmin=2 makes is a column vector

    # Add extra noise by inverting some labels at random
    rand_idx = np.random.randint(num_points,
                                 size=int(num_points * wrong_label_ratio))
    Y[:, rand_idx] = ~Y[:, rand_idx]
    Y = Y.astype(np.float)

    return X, Y


def create_train_test_split(train_to_test_ratio=0.8):
    pass


def draw_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
