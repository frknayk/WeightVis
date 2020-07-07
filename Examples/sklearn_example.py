import numpy as np
from sklearn.neural_network import MLPRegressor as MLP


from Visualizer.Brain import Brain 


X_train = np.random.rand(2,2)
y_train = np.random.rand(2,)

my_hidden_layer_sizes = (4,4)
XOR_MLP = MLP(
    activation='tanh',
    alpha=0.99,
    batch_size='auto',
    beta_1=0.9,
    beta_2=0.999,
    early_stopping=False,
    epsilon=1e-08,
    hidden_layer_sizes= my_hidden_layer_sizes,
    learning_rate='constant',
    learning_rate_init = 0.1,
    max_iter=5000,
    momentum=0.5,
    nesterovs_momentum=True,
    power_t=0.5,
    random_state=0,
    shuffle=True,
    solver='sgd',
    tol=0.0001,
    validation_fraction=0.1,
    verbose=False,
    warm_start=False)
XOR_MLP.fit(X_train,y_train)
    
# Read layer weights and bias weights together
weights = XOR_MLP.coefs_
biases_weights = XOR_MLP.intercepts_

brain_MLP = Brain(weights,biases_weights)
brain_MLP.visualize(XOR_MLP.loss_,XOR_MLP.n_iter_)
