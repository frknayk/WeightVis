import tensorflow as tf
from Libraries.Reader import Reader
from Libraries.Enums import NNLibs

class Tensorflow(Reader):
    def __init__(self):
        self.all_weights = []
        self.weights_list = []
        self.biases_list = []
        self.weight_names = []
        self.bias_names = []
        self.weights_shape = []
        self.biases_shape = []

    def read(self, weights):
        """
        Read neural network weights
        Parameters
        ----------
        weights (string/list) : Either path of the model or the weights
        """
        self.load_weight(weights)
        self.get_weights()
        self.weights_shape = [layer.shape for layer in self.weights_list]
        self.biases_shape = [layer.shape for layer in self.biases_list]

    def load_weight(self, weights=None):
        """Get weights from list of layers """
        # First, reset for live plotting.
        self.weights_list = []
        self.biases_list = []
        self.weight_names = []
        self.bias_names = []

        if type(weights) is str:
            model = tf.keras.models.load_model(weights)
            self.all_weights = model.trainable_variables
        elif weights:
            self.all_weights = weights

    def get_weights(self):
        """Load weights and biases """
        for layer in self.all_weights:
            if 'bias' in layer.name:
                self.biases_list.append(layer)
                self.bias_names.append(layer)
            else:
                self.weights_list.append(layer)
                self.weight_names.append(layer)

    def get_lib(self):
        """Get enumeration of lib"""
        return NNLibs.Tensorflow