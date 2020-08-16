import numpy as np
import activations


class TestActivations:
    def setup(self):
        # 5 examples, 2-D:
        self.x = np.array([[-1., -.5, 0., .5, 1.]]*2)

    def test_ReLU_output(self):
        """
        Assert that ReLU activation function gives the correct output
        """
        correct_output = np.array([[0., 0., 0., .5, 1.]]*2)
        np.testing.assert_array_equal(activations.ReLU.forward(self.x), correct_output)

    def test_ReLU_shape(self):
        """
        Assert that activation function does not alter the shape of the input
        """
        np.testing.assert_equal(self.x.shape, activations.ReLU.forward(self.x).shape)

    def test_Sigmoid_output(self):
        """
        Assert that sigmoid activation function gives the correct output up to the 8th decimal place
        """
        correct_output = np.array([[0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858],
                                   [0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858]])
        rounded_output = np.around(activations.Sigmoid.forward(self.x), decimals=8)

        np.testing.assert_array_equal(correct_output, rounded_output)

    def test_Sigmoid_shape(self):
        """
        Assert that activation function does not alter the shape of the input
        """
        np.testing.assert_equal(self.x.shape, activations.Sigmoid.forward(self.x).shape)

    def test_Linear_output(self):
        """
        Assert that linear activation function gives the correct output
        """
        np.testing.assert_array_equal(self.x, activations.Linear.forward(self.x))

    def test_Linear_shape(self):
        """
        Assert that activation function does not alter the shape of the input
        """
        np.testing.assert_equal(self.x.shape, activations.Linear.forward(self.x).shape)
