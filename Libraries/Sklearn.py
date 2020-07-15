import numpy as np
import pickle
from Libraries.Enums import NNLibs
from Libraries.Reader import Reader
from Utils.Bcolor import Bcolors

class Sklearn(Reader):
    def __init__(self):
        self.weights_list = []
        self.biases_list = []
        self.bcolors = Bcolors()

    def read(self,weights_biases):
        """Read neural network weights"""
        self.load_weight(weights_biases)

    def load_weight(self,weights_biases):
        """Get weights from list of layers """
        self.weights_list = weights_biases[0]
        self.biases_list = weights_biases[1]

    def get_weights(self,path):
        """Load weights and biases """
        file_name = path + ".pickle"
        with open(file_name, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def get_lib(self):
        """Get enumeration of lib"""
        pass
        