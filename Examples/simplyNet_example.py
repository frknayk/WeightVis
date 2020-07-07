from Libraries.Simplynet import Read_SimplyNet
from Visualizer.Brain import Brain


# Read SimplyNet weights
simple_weights = Read_SimplyNet("Models/fcnn_weight_3_2526")


# Initate visualizer
brain = Brain(simple_weights.weights_list, simple_weights.biases_list)

# Visualize neural network
brain.visualize()
