from Visualizer.Brain import Brain
from Libraries.Enums import NNLibs as Libs

# Initate visualizer
brain = Brain(nn_lib=Libs.Torch)

# Visualize neural network
brain.visualize("Models/sample_5",load_from_path=True)