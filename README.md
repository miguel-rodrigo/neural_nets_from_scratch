# neural_nets_from_scratch
The purpose of this project is purely educational. Is not meant to be eficient or practical in any other way

## How to use it
Create a model object providing the following parameters
  - n_units_vector: a list-like object with the structure of [input_size, hidden_layer1_size, hidden_layer2_size, ..., hidden_layern_size]. The last hidden layer refers to the output, so its size and the output size should match. This list will have length equal to the number of layers + 1 for the input layer.
  - activations: a list-like object of length equal to the number of layers (in other words, one less than the list above). Each element of this list must be a subclass of Activation class contained in the activations.py module. Current options are activations.ReLU, activations.Sigmoid and activations.Linear.
  - (optional) parameters: if provided, it will initialize the network with the parameters specified. It must be a list of dictionaries. Each dictionary will have a "W" and a "b" element, and they will override the random parameters on each layer. Needlessly to say, this list of dictionaries must have the same length as the number of layers in the model.

Once you have the model, you can call its train() method, supplying an X and a Y matrices, which contain the observations and labels respectively. Observations must be organized column-wise, that is, each column is a different example, while each row is a variable of the same observation. In the same way, Y will have each different label in a column, and only multi-dimensional outputs will have more than one row.

After being trained, you can make inferences calling the model's predict() method, and providing a new X as an argument. This new X can have more than one observation, in which case the output Y_hat will have more than one column.


## Known issues
Backprop is currently broken and a fix is incoming
