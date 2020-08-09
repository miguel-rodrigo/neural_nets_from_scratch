import numpy as np


def create_fake_data(num_points=200, num_petals=6):
    assert num_petals % 2 == 0, "# of petals has to be an even number"

    np.random.seed(123)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    r = np.cos(num_petals//2)

    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)

    # TODO: Add random jitter to X
    X = np.vstack((x1, x2))

    # Octant I: 0
    # Octants II, III: 1
    # Octant IV: 0
    # Octant V: 1
    # Octatns VI, VII: 0
    # Octact VIII: 1
    octants = np.floor(theta / (2*np.pi) * 8) + 1
    zero_octants = [1, 4, 6, 7]
    Y = [0 if octant in zero_octants else 1 for octant in octants]

    return X, Y


def create_train_test_split(train_to_test_ratio=0.8):
    pass


def draw_decision_boundary(model):
    pass
