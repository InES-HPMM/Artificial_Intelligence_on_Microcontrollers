""" Script to evaluate the converted neural networks on the STM32F29

Sends mnist images to the STM32F29 where a neural network runs and predicts the digits.
Predictions are compared to true labels.

Example use:
    python3 eval.py /dev/ttyUSB0 100

:Params
    - /dev/ttyUSB0 name of serial device (see M4Driver.py for more information)
    - 100 number of test images to evaluate the neural net on

:Author: Raphael Zingg zing@zhaw.ch
:Copyright: 2020 ZHAW / Institute of Embedded Systems
"""
import sys
import struct
import tensorflow as tf
from M4Driver import M4Driver

# -------------------------------------------------------------------------------------------------
# Get parameters from command line
# -------------------------------------------------------------------------------------------------
SER_DEV = str(sys.argv[1])
NUM_TEST = int(sys.argv[2])

# -------------------------------------------------------------------------------------------------
# Get data, only test set is required
# -------------------------------------------------------------------------------------------------
(_, _), (X_TEST, Y_TEST) = tf.keras.datasets.mnist.load_data()

# -------------------------------------------------------------------------------------------------
# Open serial connection to the M4 board
# -------------------------------------------------------------------------------------------------
m4d = M4Driver()
fw = m4d.openSerial(SER_DEV, baud=256000)

# Validate firmware, its hardcoded in the main source file
if fw == 1:
    rb_fw = 'renesas'
elif fw == 2:
    rb_fw = 'st'
elif fw == 3:
    rb_fw = 'tfLite'
elif fw == 4:
    rb_fw = 'nnom'

# -------------------------------------------------------------------------------------------------
# Get the predictions from the board
# -------------------------------------------------------------------------------------------------
print('\n\nFirmware running on target:' + rb_fw + ' evaluate:' + str(NUM_TEST) + ' samples!\n\n')
wrong_pred = 0
target_pred = []
for i in range(0, NUM_TEST):

    # get prediction
    ret = m4d.predict(X_TEST[i].reshape(1, 28*28))
    target_pred.append(struct.unpack('1B', ret)[0])
    print(str(i) + ' Target:' + str(target_pred[-1]) + ' Label:' + str(Y_TEST[i]))

    # store if target prediction is not correct
    if target_pred[-1] != Y_TEST[i]:
        wrong_pred = wrong_pred + 1

print('\n\nAcc target:', 100 - (wrong_pred / i) * 100)
