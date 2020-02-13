# Torcs Environment
import numpy as np
import math
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Brain = torch.load( "Best_Actor_Weights/actormodel.pth" )

"""
Uncomment those to see what is like a torch neural network 
"""
# print(Brain)
# print(Brain['acceleration.weight'])

for layer in Brain : 
  print(layer)
  print("********")

"""
############## TODO ##########
 1. Detect .bias and .weight automatically
 2. Store .bias and .weight in different dictionaries 
 3. Visaulize layers (not biases just weights) by MATPLOTLIB
"""
