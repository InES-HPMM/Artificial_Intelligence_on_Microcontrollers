# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import os
import pickle as pkl
import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf

BST_PATH = 'BSR_bsds500.tgz'
size28 = True

rand = np.random.RandomState(42)

f = tarfile.open(BST_PATH)
train_files = []
for name in f.getnames():
    if name.startswith('BSR/BSDS500/data/images/train/'):
        train_files.append(name)

print('Loading BSR training images')
background_data = []
for name in train_files:
    try:
        fp = f.extractfile(name)
        bg_img = skimage.io.imread(fp)
        background_data.append(bg_img)
    except:
        continue


def compose_image(digit, background):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    
    bg = background[x:x+dw, y:y+dh]
    return np.abs(bg - digit).astype(np.uint8)


def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    if size28:
        d = x.reshape([28, 28, 1]) * 255
    else:
        d = x.reshape([32, 32, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    if size28:
        X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    else:
        X_ = np.zeros([X.shape[0], 32, 32, 3], np.uint8)
    for i in range(X.shape[0]):

        if i % 1000 == 0:
            print('Processing example', i)

        bg_img = rand.choice(background_data)

        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d

    return X_
if size28:
    (mnist_train, train_labels), (mnist_test, test_labels) = tf.keras.datasets.mnist.load_data()
else:
    mnist_train = np.reshape(np.load('train_mnist_32x32.npy'), (55000, 32, 32, 1))
    mnist_train = np.reshape(mnist_train, (55000, 32, 32, 1))
    mnist_train = mnist_train.astype(np.float32)
    
    mnist_test = np.reshape(np.load('test_mnist_32x32.npy'), (10000, 32, 32, 1))
    mnist_test = np.reshape(mnist_test, (10000, 32, 32, 1))
    mnist_test = mnist_test.astype(np.float32)

print('Building train set...')
train = create_mnistm(mnist_train)

print('Building test set...')
test = create_mnistm(mnist_test)


# Save dataset as pickle
if size28:
    with open('mnistm_data28.pkl', 'wb') as f:
        pkl.dump({ 'train': train, 'test': test}, f, pkl.HIGHEST_PROTOCOL)
else:
    with open('mnistm_data32.pkl', 'wb') as f:
        pkl.dump({ 'train': train, 'test': test}, f, pkl.HIGHEST_PROTOCOL)
