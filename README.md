# WeightVis
Visualize a trained neural network weights from different libraries

## Table of Contents

- [About](#about)
- [Intallation Started](#getting_started)
- [TODO](#todo)
- [Contributing](#contributing)
- [Authors](#authors)
- [Acknowledgements](#acknowledgement)

## About <a name = "about"></a>

- Load neural network weights, WeightVis will automatically visualize the neural network weights ! 
- For now the library works with only for fully connected layers
- Supported Libraries : Pytorch, Sklearn and SimplyNet(https://github.com/frknayk/SimplyNet)
- Tensorflow will be integrated soon !

## Usage <a name = "usage"></a>

- Simplest example with pytorch

```
from Visualizer.Brain import Brain
from Libraries.Enums import NNLibs as Libs

# Initate visualizer
brain = Brain(nn_lib=Libs.Torch)

# Visualize neural network
brain.visualize("path_your_pytorch_model", load_from_path=True)
```

- Output

<img width=640px height=480px src="images\pytorch_output.png" alt="Project logo">

## Installation <a name = "getting_started"></a>

- pip3 install -e .

## TODO <a name = "todo"></a>

1. Tensorflow is incoming !
2. Extend library to CNNs 
3. Arise warning when neura-network size exceeds limits of library 
4. Long term TODO : 3D visualisation

## Contribution <a name = "contributing"></a>
- Please send an email to furkanayik@outlook.com or abdu.kaan@gmail.com for contribution or any feedback

## ✍️ Authors <a name = "authors"></a>

- [@AbdKaan](https://github.com/AbdKaan) - Idea & developer
- [@frknayk](https://github.com/frknayk) - Idea & developer

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- The script was first moved from a github user's [@craffel] "draw_neural_net.py" script.
