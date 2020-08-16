import numpy as np


class TestUtils:
    def setup(self):
        """
        Create a fake model object, which has all that is needed (parameters, gradients and cost function) but
        everything has been manually calculated on paper.

        Then try a case when the gradients are correct, and a case when they are not.
        """

    def test_gradient_checking(self):
        np.testing.assert_equal(True, False)
