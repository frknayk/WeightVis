import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from Visualizer.Brain import Brain
from Libraries.Enums import NNLibs as Libs
'''
# Load and prepare mnist dataset.
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.reshape(x_train, [60000, -1])
x_test = np.reshape(x_test, [10000, -1])
'''

# Smaller random dataset for live plotting
x_train = tf.random.normal([10000, 32], 0, 1, tf.float64)
y_train = tf.random.normal([10000, 1], 5, 1, tf.float64)

x_test = tf.random.normal([1000, 32], 0, 1, tf.float64)
y_test = tf.random.normal([1000, 1], 5, 1, tf.float64)

# Use tf.data to batch and shuffle the dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Build the model
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(8, activation='relu')
        self.d2 = Dense(4, activation='relu')
        self.d3 = Dense(10)
        
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

model = MyModel()

# Create optimizer and loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

# Metrics to measure loss/accuracy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Use tf.GradientTape to train the model
@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# Test the model
@tf.function
def test_step(data, labels):
    predictions = model(data, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

# Initiate Visualizer
weightVis = Brain(nn_lib=Libs.Tensorflow)

# Training
EPOCHS = 500
visualize_interval = 10

for epoch in range(EPOCHS):
    # Reset the metrics at the start of every epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for data, labels in train_ds:
        train_step(data, labels)

    for test_data, test_labels in test_ds:
        test_step(test_data, test_labels)
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))

    if epoch % visualize_interval == 0:
        # Plot
        weightVis.visualize(model.trainable_variables, loss_=test_loss.result(), n_iter_=epoch, interval=1)


# Save Model
#model.save('tf_sample')