"""
Script to convert a keras models into an tflite model[1].
The converted tfLite model is evaluated on test data.

:Author: Raphael Zingg zing@zhaw.ch
:Copyright: 2020 ZHAW / Institute of Embedded Systems
:References:
 - [1] TensorFlow, Get started with microcontrollers,
   https://www.tensorflow.org/lite/microcontrollers/get_started
"""
import numpy as np
import tensorflow as tf

# -------------------------------------------------------------------------------------------------
# Settings / Constants
# -------------------------------------------------------------------------------------------------
img_rows, img_cols = 28, 28

# -------------------------------------------------------------------------------------------------
# Get the trained reference  model
# -------------------------------------------------------------------------------------------------
(x_train_int, _), (x_test_int, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train_int / 255.0, x_test_int / 255.0
keras_file = '../neural_networks/mnist_cnn.h5'
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)

# -------------------------------------------------------------------------------------------------
# Quantize, convert and store the model
# -------------------------------------------------------------------------------------------------
converter.inference_type = tf.uint8
converter.inference_input_type = tf.uint8

# dict of strings representing input tensor names mapped to tuple of floats representing the
# mean and standard deviation of the training data (e.g., {"foo" : (0., 1.)})
converter.quantized_input_stats = {"conv2d_input": (np.mean(x_train), np.std(x_train))}

# tuple of integers representing (min, max) range values for all arrays without a specified range.
# Intended for experimenting with quantization via "dummy quantization". (default None)
converter.default_ranges_stats = [-255*255, 255*255]

# convert it
tflite_model_quant = converter.convert()

# save the converted model
tflite_model_quant_file = 'mnist_model_tflite.tflite'
open(tflite_model_quant_file, "wb").write(tflite_model_quant)

# -------------------------------------------------------------------------------------------------
# Evaluate the converted tflite model, compare with original model
# -------------------------------------------------------------------------------------------------
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter.allocate_tensors()
error = 0
i = 0
for i, img in enumerate(x_test_int):

    example_img_for_tflite = np.reshape(img, (1, 28, 28, 1)).astype(np.uint8)

    # put data into the model
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], example_img_for_tflite)

    # run model
    interpreter.invoke()

    # get output of the model
    output_data = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])

    # check if prediction was right, sum up wrong predictions
    pred = np.argmax(output_data)
    if pred != y_test[i]:
        error = error + 1

print("Acc converted tfLite model:", 100 - error/len(x_test_int)*100)
