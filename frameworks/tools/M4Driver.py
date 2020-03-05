# pylint: disable=C0103
""" Driver class to interact with STM32F29-Disco through UART

Uses serial UART connection to exchange data and commands
Hardware: https://ennis.zhaw.ch/wiki/doku.php based on STM32F29-Disco
Connect:
    PA0-WKUP Board ------> TX Serial device host
    PA1      Board ------> RX Serial device host
    GND      Board ------> GND host

:Author: Raphael Zingg zing@zhaw.ch
:Copyright: 2020 ZHAW / Institute of Embedded Systems
"""

import struct
import serial as ser


class M4Driver:

    def __init__(self):
        self.ser = []

    def openSerial(self, serial_port, baud=256000):
        """Open a serial connection and perform a handshake

        serialPort: (str) of the serial device eg: COM3 or /dev/ttyUSB0
        returns: string of firmware running on the device
        """
        # open serial port and set properties
        try:
            self.ser = ser.Serial(serial_port, baudrate=baud, timeout=10)
        except ser.SerialException:
            print("SerialException, cant open port!")

        # hardcoded handshake: write 's' receive 'X'
        try:
            self.ser.write(b's')
            ret = self.ser.read(1)
            if ret.decode("utf-8")[0] == 'X':
                print('Connection open!')
                ret = self.ser.read(1)
                return ret.decode("utf-8")[0]
        except IndexError:
            print('Handshake failed, check if firmware is running!')

    def predict(self, image):
        """ Returns prediction of the mnist image mnistIntImage

        image:  (np.array[1][28*28]) with a MNIST image
        returns: prediction read from the serial device
        """
        prediction = []

        # write command c and send the data
        self.ser.write(b'c')
        for i in range(0, 28*28):
            self.ser.write(struct.pack('!B', image[0][i]))

        # get the prediction
        prediction = self.ser.read(1)
        if prediction != []:
            return prediction
