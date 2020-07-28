from Visualizer.Brain import Brain
from Libraries.Enums import NNLibs as Libs

# Initate visualizer
brain = Brain(nn_lib=Libs.Tensorflow)

# Visualize neural network
brain.visualize("Models/tf_sample", load_from_path=True)