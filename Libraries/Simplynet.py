import numpy as np
import pickle

class Read_SimplyNet:
    def __init__(self,path):
        self.weights_list = []
        self.biases_list = []
        self.load_weights(path)

    def get_weights(self,path):
        """Load weights and biases """
        file_name = path + ".pickle"
        with open(file_name, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def load_weights(self,path):
        """Get weights from list of layers """
        weights_list = self.get_weights(path)
        for idx in range(len(weights_list)):
            # Layers are stored as dict where layer name is key in SimplyNet
            layer_dict = weights_list[idx]
            
            # Get the key (there is single key : layer name!)
            key_list = [thing for thing in layer_dict.keys()]

            # Get weights dictionary
            weigts_dict = layer_dict[key_list[0]]
            
            # Read weights and biases
            W = weigts_dict['W'].T
            b = weigts_dict['b'].T
            # bias must be in the shape of (dim_bias,)
            b = b.reshape(b.shape[1])

            self.weights_list.append(W)
            self.biases_list.append(b)
