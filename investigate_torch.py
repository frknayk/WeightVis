# %%
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Brain = torch.load( "./actormodel.pth" )

weight_names = []
bias_names = []
weights = []
biases = []

for layer in Brain:
  if 'weight' in layer:
    weights.append(Brain[layer])
    weight_names.append(layer)
  elif 'bias' in layer:
    biases.append(Brain[layer])
    bias_names.append(layer)

weights_shape = [layer.shape for layer in weights]
bias_shape = [layer.shape for layer in biases]

# print(weights_shape)
# print("Weight_names:\n", weight_names)
# counter = 0
# for layer in weights:
#   counter += 1
# print(counter)

first_layer_weight = weights[0]

def tensor_to_numpy( torch_tensor ):
  # TODO : HOW TO CONVERT GPU TO NUMPY DIRECTLY
  return torch_tensor.cpu().numpy()

np_arr = tensor_to_numpy(first_layer_weight)

print(np_arr.shape)

'''
print("Weight_names:\n", weight_names)
print("Bias_names:\n", bias_names)

print("Weights:\n", weights)
print("Biases:\n", biases)
'''

"""
############## TODO ##########
 1. Visaulize layers (not biases just weights) by MATPLOTLIB

Reference : https://www.youtube.com/watch?v=3JQ3hYko51Y
"""

