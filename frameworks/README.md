## Embedded Neural Network Framework Evaluation
This repository contains embedded artificial intelligence applications for the [STM32F429 discovery board](https://www.st.com/en/evaluation-tools/32f429idiscovery.html),
 generated with different embedded artificial intelligence frameworks:
 - [STMicroelectronics: X-CUBE-AI (cube)](cube)
 - [Jianjia Ma: Neural Network on Microcontroller with CMSIS-NN backend from ARM (nnom)](nnom)
 - [Renesas: e-AI Translator (e-Ai)](e_ai)
 - [Google: Tensorflow Lite for Microcontrollers(tfLite)](tfLite)

## Requirements

### Hardware
 - [STM32F429 Discovery board](https://www.st.com/en/evaluation-tools/32f429idiscovery.html)
 - [Serial Connector](http://www.farnell.com/datasheets/814049.pdf)
 - Host Computer

### Software
 - Ubuntu 18.04 with GNU/Linux 4.15.0-70-generic
 - arm-none-eabi-gcc 6.3.1
 - python 3.7.3 with sklearn, matlabplotib and pyserial
 - tensorflow 1.14.0
 - keras 2.2.4
 - gcc 7.4.0
 - arm-none-eabi-gcc 6.3.1
 - [stlink](https://github.com/texane/stlink) 1.5.1
 - [STM32CubeMX](https://www.st.com/en/development-tools/stm32cubemx.html) 5.4.0
 - [X-CUBE-AI](https://www.st.com/en/embedded-software/x-cube-ai.html) 4.1.0
 - [e-ai](https://www.st.com/en/embedded-software/x-cube-ai.html) 1.0.0

 ## Preparation
 ### Build Stlink
 ```bash
$ cd tools/stlink
$ make
```
