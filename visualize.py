import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor as MLP
from draw_net import draw_neural_net

# X_train = np.random.rand(4,2)
# y_train = np.dot(X_train, np.random.rand(2,1) )+ 3

X_train = np.random.rand(2,2)
y_train = np.random.rand(2,)

print(X_train.shape)
print(y_train.shape)

my_hidden_layer_sizes = (2,)


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
#----------[2-2] Training
XOR_MLP.fit(X_train,y_train)

# Plot the Neural Network
fig66 = plt.figure(figsize=(12, 12))
ax = fig66.gca()
ax.axis('off')

draw_neural_net(ax, .1, .9, .1, .9, [2,2,1],XOR_MLP.coefs_,XOR_MLP.intercepts_,XOR_MLP.n_iter_,XOR_MLP.loss_,np, plt)
plt.show()