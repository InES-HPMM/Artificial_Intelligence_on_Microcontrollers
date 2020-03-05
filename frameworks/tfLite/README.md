

## Workflow with [Tensorflow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
### Convert Neural Network into Source Code
Generate the weights for the dense model with:
```bash
$ python3 generateDense.py
```
or the weights for the cnn model with:
```bash
$ python3 generateCnn.py
```
Finally, convert tfLite model to a .cc file:
```bash
$ xxd -i mnist_model_tflite.tflite > ./tensorflow/lite/micro/examples/mnist/model_data.cc
```

### Build and Flash
Bbuild the firmware and finally flash it:
```bash
$ make
$ ../tools/stlink/build/Release/st-flash --format ihex write ./build/TFLIT.hex
```
### Evaluate the tfLite for Microcontrollers Neural Network on the STM32F429

#### Memory
To analyse the memory footprint of the firmware you can use:
```bash
$ arm-none-eabi-size build/TFLIT.hex 
```
The output should look like this:

![size output](../../doc/arm-none-eabi-size_output.png)

To analyse the memory required for the neural network you can look at the size of:
`/tensorflow/lite/micro/examples/mnist/model_data.cc` with:
```bash
$ ll ./tensorflow/lite/micro/examples/mnist/model_data.cc
```
The output should look like this:

![ll output](../../doc/output_ll.png)

#### Accuracy
Connect the serial device hardware to your serial device
```
PA0-WKUP STM32F429-Board ------> TX Host serial device 
PA1      STM32f429-Board ------> RX Host derial device
GND      STM32f429-Board ------> GND Host
```
Get the name of your serial connection:
```
$ dmesg
```
The output should look like this:

![dmesg output](../../doc/dmesg_output.png)

The last part of the message, here ttyUSB0, is the name of your serial device.

Switch into the tool directory and run the evaluation script.
**!CAUTION: Adjust the parameter `/dev/ttyUSB0`!**
```bash
$ cd ../tools/
$ python3 eval.py /dev/ttyUSB0 100
```
The command above:
`python3 eval.py /dev/ttyUSB0 100`
sends `100` test images to the STM32F429 and evaluates the predictions from the board.
Maximum of `10000` images can be evaluated.

#### Runtime
To measure the inference runtime of the neural net connect the GPIO `PIN0` of `GPIOB` to an oscilloscope
The output of your scope should look like this:

![dmesg output](../../doc/scope_output.png)
