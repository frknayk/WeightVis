# Torcs Environment
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

# Neural network weights and layer names
def get_Weights(NN_Brain):
  weight_names = []
  weights = []
  for layer in Brain: 
    if 'weight' in layer:
      weights.append(Brain[layer])
      weight_names.append(layer)
  
  return weights,weight_names

# Neural network weights' biases and bias names
def get_Biases(NN_Brain):
  biases = []
  bias_names = []
  for layer in Brain: 
    if 'bias' in layer:
      biases.append(Brain[layer])
      bias_names.append(layer)
  return biases,bias_names

# Number of layers and how many neurons in each layers ?
def get_Shape(NN_weights):
  number_of_layers = len(brain_weights)
  layers_size = []
  for x in range(number_of_layers):
    layer_x = NN_weights[x]
    print(layer_x.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Brain = torch.load( "./actormodel.pth" )

# brain_weights,brain_weights_names = get_Weights(Brain)
# brain_biases,brain_biases_names = get_Biases(Brain)

# # get_Shape(brain_weights)
# print(Brain.items())
brain_items = Brain.items()

for k,v in sorted(Brain.items()):
    print(k, tuple(v.shape) )


"""
############## TODO ##########
 1. Detect .bias and .weight automatically
 2. Store .bias and .weight in different dictionaries 
 3. Visaulize layers (not biases just weights) by MATPLOTLIB
"""