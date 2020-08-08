import numpy as np


def create_fake_data(num_points=200, num_petals=6):
    assert num_petals % 2 == 0, "# of petals has to be an even number"

    np.random.seed(123)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    r = np.cos(num_petals//2)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def create_train_test_split(train_to_test_ratio=0.8):
    pass


def draw_decision_boundary(model):
    pass
