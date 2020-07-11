from abc import ABC, abstractmethod

class Reader(ABC):
    def read(self):
        """Read neural network weights"""
        pass

    def load_weight(self):
        """Get weights from list of layers """
        pass

    def get_weights(self):
        """Load weights and biases """
        pass

    def get_lib(self):
        """Get enumeration of lib"""
        pass
