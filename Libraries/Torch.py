import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from Utils.Bcolor import Bcolors
from Utils.Debug_Levels import Debug_Levels as verbos

class Read_Torch:
  def __init__(self,weight_path=None,trained_weights=None,debug_level = verbos.NO):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.brain_path = weight_path
    self.trained_weights = trained_weights  
    self.verbos_level = debug_level
    self.weights_all = None
    self.weight_names  = []
    self.bias_names    = []
    self.weights       = []
    self.weights_shape = None
    self.biases        = []
    self.biases_shape = None
    self.bcolors = Bcolors()
    # Brain Vis wants all weights in this format
    self.weights_list  = []
    self.biases_list   = []

    self.load_weight()
    self.get_weights()
    self.print_layer_info()
    self.make_list_of_layers()
    self.transpose_layers()
    self.weights_shape    = [layer.shape for layer in self.weights_list]
    self.biases_shape     = [layer.shape for layer in self.biases_list]


  def load_weight(self):
    if self.brain_path is not None:
      try:
        self.weights_all = torch.load(self.brain_path)
      except:
        self.bcolors.print_error("Neural network weights could not be loaded. Please check the path !")
    else:
      try:
        self.weights_all = self.trained_weights
      except:
        self.bcolors.print_error("Trained weights could not be read !")

  def get_weights(self):
    Brain = self.weights_all
    for layer in Brain:
      if 'weight' in layer:
        self.weights.append(Brain[layer])
        self.weight_names.append(layer)
      elif 'bias' in layer:
        self.biases.append(Brain[layer])
        self.bias_names.append(layer)

  def make_list_of_layers(self):
    """
    @brief Read layers weights and return
    """
    for weight in self.weights:
      weight_np = self.tensor_to_numpy(weight)
      self.weights_list.append(weight_np)
  
    for bias in self.biases:
      bias_np = self.tensor_to_numpy(bias)
      self.biases_list.append(bias_np)
    
  def tensor_to_numpy(self, torch_tensor ):
    return torch_tensor.cpu().numpy()

  def print_layer_info(self):
    if self.verbos_level.value > verbos.MEDIUM.value:
      self.bcolors.print_ok("Node weights and Bias weights are seperated successfully !")
      self.bcolors.print_bold("NODE WEIGHTS")

      for node_weight_name in self.weight_names:
        self.bcolors.print_underline(node_weight_name)

      self.bcolors.print_bold("BIAS WEIGHTS")
      for bias_weight_name in self.bias_names:
        self.bcolors.print_underline(bias_weight_name)
      print("\n************************")

  def get_layer_shapes(self,calculate=False):
    if self.weights_shape is None:
      calculate = False
    if self.biases_shape is None:
      calculate = False
    if(self.weights_shape is not None) and (self.biases_shape is not None) and (calculate):
      self.weights_shape    = [layer_weights.shape for layer_weights in self.weights_list]
      self.biases_shape     = [layer_biases.shape  for layer_biases  in self.biases_list]

    if self.verbos_level.value > verbos.MEDIUM.value:
      self.bcolors.print_inform("... Node Weights Layer Shapes ...")
      print(self.weights_shape)
      self.bcolors.print_inform("... Bias Weights Layer Shapes ...")
      print(self.biases_shape)

  def transpose_layers(self):
    weights_list_transposed  = []
    biases_list_transposed   = []

    for weight in self.weights_list:
      weights_list_transposed.append(weight.transpose() )

    for weight in self.biases_list:
      biases_list_transposed.append(weight.transpose() )
    
    self.weights_list = weights_list_transposed
    self.biases_list  = biases_list_transposed
