# import numpy as np
#
# import utils
# import activations
# from model import Model
#
#
# class TestUtils:
#     def setup(self):
#         np.random.seed(1)
#         n_units = [2, 4, 1]
#
#         activations = [activations.ReLU, activations.Sigmoid]
#         self.model = Model(n_units, activations)
#
#         X, Y = utils.create_fake_data()
#         n_epochs = 1000
#         for i, iteration_loss in enumerate(self.model.train(X, Y, n_epochs=n_epochs)):
#             if i % 100 == 0 or i == n_epochs - 1:
#                 print("Loss on iteration {}: {}".format(i, iteration_loss))
#
#     def test_gradient_checking(self):
#         self.assertEqual(True, False)
