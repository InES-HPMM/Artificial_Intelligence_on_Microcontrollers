"""
Script to convert a keras model into an optimized NNOM[1] model using CMSIS-NN[2] as backend.

:Author: Raphael Zingg zing@zhaw.ch
:Copyright: 2020 ZHAW / Institute of Embedded Systems
:References:
 - [1] Jianjia Ma, Neural Network on Microcontroller (NNoM), https://github.com/majianjia/nnom
 - [2] ARM, CMSIS, https://github.com/ARM-software/CMSIS_5
"""
import tensorflow as tf
import keras as k
from nnom_utils import generate_model

# -------------------------------------------------------------------------------------------------
# Settings / Constants
# -------------------------------------------------------------------------------------------------
img_rows, img_cols = 28, 28

# -------------------------------------------------------------------------------------------------
# Data required for quantization
# -------------------------------------------------------------------------------------------------
(_, _), (x_test_int, _) = k.datasets.mnist.load_data()
x_test = x_test_int / 255.0
x_test = x_test.reshape(x_test.shape[0], 28*28)

# -------------------------------------------------------------------------------------------------
# Define keras model with syntax that nnom can parse
# -------------------------------------------------------------------------------------------------
x = k.layers.Input(shape=x_test.shape[1:])
fc = k.layers.Dense(10)(x)
y = k.layers.Softmax()(fc)
model = k.models.Model(inputs=x, outputs=y)

# get the trained weights
keras_file = '../neural_networks/mnist_min.h5'

trained_model = tf.keras.models.load_model(keras_file)
model.set_weights(trained_model.get_weights())

# -------------------------------------------------------------------------------------------------
# Convert the keras model and store it in target/Inc/weights.h
# -------------------------------------------------------------------------------------------------
generate_model(model, x_test, name='target/Inc/weights.h')
