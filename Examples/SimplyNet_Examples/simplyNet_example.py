from Libraries.Simplynet import SimplyNet
from Visualizer.Brain import Brain


# Read SimplyNet weights
simple_weights = SimplyNet("Models/random_weight_2_4_4_1")


# Initate visualizer
brain = Brain(simple_weights.weights_list, simple_weights.biases_list)

# Visualize neural network
brain.visualize()
