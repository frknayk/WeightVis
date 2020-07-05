from Libraries.Simplynet import Read_SimplyNet
from Visualizer.Brain import Brain

# Absolute path to neural network weights
path = "/home/furkan/Furkan/Codes/Brain_Visualizer/Models/simply_net_data"

# Read SimplyNet weights
simple_weights = Read_SimplyNet(path)

# Initate visualizer
brain = Brain(simple_weights.weights_list, simple_weights.biases_list)

# Visualize neural network
brain.visualize()
