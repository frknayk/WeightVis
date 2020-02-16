# Torcs Environment
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

print(weights_shape)
print(bias_shape)

'''
print("Weight_names:\n", weight_names)
print("Bias_names:\n", bias_names)

print("Weights:\n", weights)
print("Biases:\n", biases)
'''

"""
############## TODO ##########
<<<<<<< HEAD:investigate_torch_old.py
 1. Detect .bias and .weight automatically
 2. Store .bias and .weight in different dictionaries 
 3. Visaulize layers (not biases just weights) by MATPLOTLIB
"""
=======
 1. Visaulize layers (not biases just weights) by MATPLOTLIB
"""
>>>>>>> b2511d29de9feb46148ec02e04c9cce9588eb418:investigate_torch.py
