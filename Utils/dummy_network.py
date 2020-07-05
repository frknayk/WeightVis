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
    def __init__(self, num_inputs, hidden_size1, hidden_size2,init_w=3e-3):

        super(PolicyNetwork, self).__init__()

        # 2 Hidden Layer
        self.linear1 = nn.Linear(num_inputs, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)

        # Final Layer
        self.output = nn.Linear(hidden_size2,1)

if __name__ == "__main__":
    # Initialize model
    model = PolicyNetwork(2,4,4)
    torch.save(model.state_dict(),'sample_2')
