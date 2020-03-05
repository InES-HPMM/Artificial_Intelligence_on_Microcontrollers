"""
Script to train and store some MNIST keras models

Those models are just used as dummy models to show how the different embedded frameworks
work and not to achieve high accuracy.

:Author: Raphael Zingg zing@zhaw.ch
:Copyright: 2020 ZHAW / Institute of Embedded Systems
"""

import tensorflow as tf

# -------------------------------------------------------------------------------------------------
# Settings / Constants
# -------------------------------------------------------------------------------------------------
img_rows, img_cols = 28, 28

# -------------------------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------------------------
(x_train_int, y_train), (x_test_int, y_test) =  tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train_int / 255.0, x_test_int / 255.0
x_train = x_train.reshape(x_train.shape[0], img_rows*img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows*img_cols)

# -------------------------------------------------------------------------------------------------
# Create a dense keras model and train on mnist data, save it
# -------------------------------------------------------------------------------------------------
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, input_shape=(img_rows*img_cols,), activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=1)

# Save the model
keras_file = 'mnist_min.h5'
tf.keras.models.save_model(model, keras_file)

# -------------------------------------------------------------------------------------------------
# Create a cnn keras model and train on mnist data, save it
# -------------------------------------------------------------------------------------------------
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                 activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=1)

keras_file = 'mnist_cnn.h5'
tf.keras.models.save_model(model, keras_file)
