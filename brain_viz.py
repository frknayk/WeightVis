import numpy as np
import matplotlib.pyplot as plt
from utils import Bcolors
from math import fabs

#TODO give credit to asian
#TODO add verbosity levels for Bcolors 
#TODO make docstring like numpy

class Brain:
    """
    Draw a neural network graph using matplotlib (for now)
    
    parameters : 
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - fig_size :
            Figure size in the X,Y directions
        - offset_left : float
            The center of the leftmost node(s) will be placed here
        - offset_right : float
            The center of the rightmost node(s) will be placed here
        - offset_bottom : float
            The center of the bottommost node(s) will be placed here
        - offset_top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
        - weights :(list) length (n_layers - 1) The ith element in the list represents the weight matrix corresponding to layer i.
        - bias_weights : (list) length (n_layers - 1)The ith element in the list represents the bias vector corresponding to layer i + 1.
        - n_iter: (int) The number of iterations the solver has ran.
        - loss: (float) The current loss computed with the loss function.
    """

    def __init__(self, nn_weights, nn_bias_weights, fig_size_x=12, fig_size_y=12):
        '''
        Initialize brain.

        Parameters
        ----------
        nn_weights : list
            Weights of the model.

        nn_bias_weights : list
            Biases of the model.
        '''
        self.layer_sizes    = None
        self.ax             = None
        
        # Figure sizes in percentage
        #TODO When figure size increases, nodes become squished.
        self.fig_size_x     = fig_size_x
        self.fig_size_y     = fig_size_y

        self.left           = 0.1
        self.right          = 0.9
        self.bottom         = 0.1
        self.top            = 0.9

        self.offset_all     = [self.left, self.right, self.bottom, self.top]
        self.weights        = nn_weights
        self.bias_weights   = nn_bias_weights

        self.n_layers       = None
        self.v_spacing      = None
        self.h_spacing      = None

        # Beautiful printing
        self.bcolors = Bcolors()

    def visualize(self,loss_ = 999,n_iter_ = 1):
        '''
        Plot everything(nodes, edges, arrows, input, output)

        Parameters
        ----------
        loss_ : int
            Brain's latest loss value.

        n_iter_: int
            Number of iterations/epochs.
        '''
        self.init_graph()
        self.set_figure()
        self.plot_input_arrows()
        self.plot_nodes()
        self.plot_bias_nodes()
        self.plot_edge_node_connections()
        self.plot_bias_edge_connections()
        self.plot_output_arrows(loss_,n_iter_)
        plt.show()

    def init_graph(self):
        '''
        Creates the figure to plot the brain.
        '''
        self.bcolors.print_header("Initiating visualization graphics")
        try :
            fig = plt.figure(figsize=(self.fig_size_x, self.fig_size_y))
            # Get current axes (gca)
            self.ax  = fig.gca()
            self.ax.axis('off')
            #TODO This gives error: "TypeError: 'AxesSubplot' object is not callable", fig.gca() returns AxesSubplot object
            # self.ax(autoscale=False)
        except:
            print("Graph could not be set !\nPlease enter the figure sizes correctly") 

    def set_figure(self):
        '''
        Check if there is a problem with figure offset, layer sizes to continue procedure.

        Returns
        -------
        is_figure_set : boolean
            True if the figure's offset and layer sizes are ok.
        '''
        is_figure_set = False
        is_offset_ok = self.is_fig_offsets_ok()
        is_layer_sizes_ok = self.set_layer_sizes()
        
        print("Layer Sizes : ",self.layer_sizes)
        if is_offset_ok and is_layer_sizes_ok:
            self.bcolors.print_inform("Visualization graphics are set !")
            self.n_layers  = len(self.layer_sizes)
            self.v_spacing = (self.top - self.bottom)/float(max(self.layer_sizes))
            self.h_spacing = (self.right - self.left)/float(max(self.layer_sizes) - 1)
            # print("vertical     unit spacing : ",self.v_spacing)
            # print("horizontal   unit spacing : ",self.h_spacing)

            is_figure_set = True
        return is_figure_set

    def is_fig_offsets_ok(self):
        #TODO Why? 
        '''
        Checks if figure's offsets are between (0-100).

        Returns
        -------
        is_correct: boolean
            True/False based on the condition.
        '''
        is_correct = True
        for size in self.offset_all:
            if size > 100 or size < 0 :
                is_correct = False
                break
        return is_correct

    def set_layer_sizes(self):
        '''
        Set layer sizes.

        Returns
        -------
        True/False
            If there are no layers, return False.
        '''
        self.layer_sizes = []
        num_of_layers = len(self.weights)

        for i in range(num_of_layers):
            layer_shape = self.weights[i].shape

            # First Layer
            if(i == 0):
                layer_size = layer_shape[0]
                self.layer_sizes.append(layer_size)

            # Other layers
            layer_size = layer_shape[1]
            self.layer_sizes.append(layer_size)

        if(len(self.layer_sizes) > 0):
            return True
        else:
            return False

    def plot_input_arrows(self):
        self.bcolors.print_ok("Plotting input arrows ..")  
        self.layer_top_0 = self.v_spacing*(self.layer_sizes[0] - 1)/2. + (self.top + self.bottom)/2.
        for m in range(self.layer_sizes[0]):
            arrow_posx = self.left-0.18
            arrow_posy = self.layer_top_0 - m*self.v_spacing
            # print(arrow_posx,arrow_posy)
            plt.arrow(arrow_posx,arrow_posy , 0.12, 0,  lw =1, head_width=0.01, head_length=0.02)
        
    def plot_nodes(self):
        '''
        Plot nodes, input(X1, X2,...) and output(y1, y2,...).
        '''
        self.bcolors.print_ok("Plotting nodes ..")
        for n, layer_size in enumerate(self.layer_sizes):
            layer_top = self.v_spacing*(layer_size - 1)/2. + (self.top + self.bottom)/2.
            for m in range(layer_size):
                circle = plt.Circle((n*self.h_spacing + self.left, layer_top - m*self.v_spacing), self.v_spacing/8.,\
                                color='w', ec='k', zorder=4)
                # Add texts
                if n == 0:
                    plt.text(self.left-0.125, layer_top - m*self.v_spacing, r'$X_{'+str(m+1)+'}$', fontsize=15)
                elif (self.n_layers == 3) & (n == 1):
                    plt.text(n*self.h_spacing + self.left+0.00, layer_top - m*self.v_spacing+ (self.v_spacing/8.+0.01*self.v_spacing), r'$H_{'+str(m+1)+'}$', fontsize=15)
                elif n == self.n_layers -1:
                    plt.text(n*self.h_spacing + self.left+0.10, layer_top - m*self.v_spacing, r'$y_{'+str(m+1)+'}$', fontsize=15)
                self.ax.add_artist(circle) 
    
    def plot_bias_nodes(self):
        self.bcolors.print_ok("Plotting bias nodes ..")
        # Bias-Nodes
        for n in range(len(self.layer_sizes)):
            if n < self.n_layers -1:
                x_bias = (n + 0.5) * self.h_spacing + self.left
                y_bias = self.top + 0.005
                circle = plt.Circle((x_bias, y_bias), self.v_spacing/8.,\
                                    color='w', ec='k', zorder=4)
                # Add texts
                plt.text(x_bias-(self.v_spacing/8.+0.10*self.v_spacing+0.01), y_bias, r'$1$', fontsize=15)
                self.ax.add_artist(circle)   

    def plot_edge_node_connections(self):
        self.bcolors.print_ok("Plotting edge-node connections ..")

        #TODO Make normalization dynamic
        # np_weights = np.asarray(self.weights)
        # Normalize to use for alpha(transparency)
        #normalized_weights = np_weights / np.linalg.norm(np_weights)
        # normalized_weights = (np_weights -  (-0.5093194)) / (0.494668 - (-0.5093194))

        # Edges between nodes
        for n, (layer_size_a, layer_size_b) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            layer_top_a = self.v_spacing*(layer_size_a - 1)/2. + (self.top + self.bottom)/2.
            layer_top_b = self.v_spacing*(layer_size_b - 1)/2. + (self.top + self.bottom)/2.
            
            # Normalize layer weights 
            layer_weights = self.weights[n]
            normalized_weights = layer_weights / np.linalg.norm(layer_weights)

            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    
                    # For positive values draw lines blue, for negative ones draw as red
                    # TODO : move this block to a function
                    alpha_val = None
                    if normalized_weights[m, o] < 0:
                        alpha_val = -normalized_weights[m, o]
                        color = 'r'
                    else:
                        alpha_val = normalized_weights[m, o]
                        color = 'b'

                    line = plt.Line2D([n*self.h_spacing + self.left, (n + 1)*self.h_spacing + self.left],
                                      [layer_top_a - m*self.v_spacing, layer_top_b - o*self.v_spacing], c=color, alpha=alpha_val)
                    self.ax.add_artist(line)
                    xm = (n*self.h_spacing + self.left)
                    xo = ((n + 1)*self.h_spacing + self.left)
                    ym = (layer_top_a - m*self.v_spacing)
                    yo = (layer_top_b - o*self.v_spacing)
                    rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                    rot_mo_deg = rot_mo_rad*180./np.pi
                    xm1 = xm + (self.v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                    if n == 0:
                        if yo > ym:
                            ym1 = ym + (self.v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                        else:
                            ym1 = ym + (self.v_spacing/8.+0.05)*np.sin(rot_mo_rad)
                    else:
                        if yo > ym:
                            ym1 = ym + (self.v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                        else:
                            ym1 = ym + (self.v_spacing/8.+0.04)*np.sin(rot_mo_rad)
                    plt.text( xm1, ym1,\
                             str(round(self.weights[n][m, o],4)),\
                             rotation = rot_mo_deg, \
                             fontsize = 10)
                             
    def plot_bias_edge_connections(self):
        self.bcolors.print_ok("Plotting bias-edge connections ..")
        # Edges between bias and nodes
        for n, (_, layer_size_b) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            if n < self.n_layers-1:
                #layer_top_b: y of highest node of the next layer
                layer_top_b = self.v_spacing*(layer_size_b - 1)/2. + (self.top + self.bottom)/2.
                x_bias = (n+0.5)*self.h_spacing + self.left
                y_bias = self.top + 0.005 
            for o in range(layer_size_b):
                line = plt.Line2D([x_bias, (n + 1)*self.h_spacing + self.left],
                              [y_bias, layer_top_b - o*self.v_spacing], c='k')
                self.ax.add_artist(line)
                xo = ((n + 1)*self.h_spacing + self.left)
                yo = (layer_top_b - o*self.v_spacing)
                rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
                rot_bo_deg = rot_bo_rad*180./np.pi
                xo2 = xo - (self.v_spacing/8.+0.01)*np.cos(rot_bo_rad)
                yo2 = yo - (self.v_spacing/8.+0.01)*np.sin(rot_bo_rad)
                xo1 = xo2 -0.05 *np.cos(rot_bo_rad)
                yo1 = yo2 -0.05 *np.sin(rot_bo_rad)
                plt.text( xo1, yo1,\
                     str(round(self.bias_weights[n][o],4)),\
                     rotation = rot_bo_deg, \
                     fontsize = 10)   

    def plot_output_arrows(self,loss,n_iter):
        self.bcolors.print_ok("Plotting output arrows ..")
        # Output-Arrows
        layer_top_0 = self.v_spacing*(self.layer_sizes[-1] - 1)/2. + (self.top + self.bottom)/2.
        for m in range(self.layer_sizes[-1]):
            plt.arrow(self.right+0.015, layer_top_0 - m*self.v_spacing, 0.16*self.h_spacing, 0,  lw =1, head_width=0.01, head_length=0.02)
        # Record the n_iter_ and loss
        plt.text(self.left + (self.right-self.left)/3., self.bottom - 0.005*self.v_spacing, \
                 'Steps:'+str(n_iter)+'    Loss: ' + str(round(loss, 6)), fontsize = 15)

if __name__ == "__main__":
    # Read layer weights and bias weights together
    from torch_vis import Read_Torch
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    path += "/Models/sample_2"
    torch_weights = Read_Torch(path)
    print(type(torch_weights.weights_list))
    brain_MLP = Brain(torch_weights.weights_list, torch_weights.biases_list)
    #TODO Get loss and n_iter for visualize() function
    brain_MLP.visualize()
