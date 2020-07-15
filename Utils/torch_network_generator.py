import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# Using Cuda
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Actor Network
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, output, *hidden_sizes, init_w=3e-3):

        super(PolicyNetwork, self).__init__()

        nn_length = len(hidden_sizes)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, hidden_sizes[0]))
        # Hidden Layers
        for i in range(nn_length):
            if(i != nn_length-1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        self.layers.append(nn.Linear(hidden_sizes[nn_length-1], output))

if __name__ == "__main__":
    # Initialize model
    model = PolicyNetwork(16, 1, 8, 4, 2)
    torch.save(model.state_dict(),'Models/sample_4')
