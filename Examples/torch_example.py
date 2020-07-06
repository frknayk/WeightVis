from Libraries.Torch import Read_Torch
from Visualizer.Brain import Brain

# Read weights
torch_weights = Read_Torch("Models/sample_4")

# Initate visualizer
brain = Brain(torch_weights.weights_list, torch_weights.biases_list)

# Visualize neural network
brain.visualize()
