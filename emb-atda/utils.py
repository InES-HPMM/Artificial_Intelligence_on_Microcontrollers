"""
Helper functions for embedded asymmetric tri-training for unsupervised domain adaptation (emb-atda)

:Author: Raphael Zingg zing@zhaw.ch
:Copyright: 2020 ZHAW / Institute of Embedded Systems
"""
import pickle as pk
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight

#-------------------------------------------------------------------------------------------------#
# Data                                                                                            #
#-------------------------------------------------------------------------------------------------#
def return_mnist():
    '''    
    returns: mnist data set and labels split into train/test set format (28x28x3)
    '''
    # input image dimensions
    img_rows, img_cols = 28, 28

    # load mnist
    (mnist_train, train_labels), (mnist_test, test_labels) = tf.keras.datasets.mnist.load_data()

    # reshape with channels_last as tensorflow expects it, use 3 channels same as [1]
    mnist_train = mnist_train.reshape(mnist_train.shape[0], img_rows, img_cols, 1)
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = mnist_test.reshape(mnist_test.shape[0], img_rows, img_cols, 1)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

    # scale and remove pixel mean
    pixel_mean_train = np.mean(mnist_train)
    pixel_mean_test = np.mean(mnist_test)
    mnist_train = (mnist_train - pixel_mean_train) / 255
    mnist_test = (mnist_test - pixel_mean_test) / 255  

    train_labels_hot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels_hot = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    return mnist_train, mnist_test, train_labels_hot, test_labels_hot

def return_mnistm():
    '''    
    returns: mnistm data set and labels split into train/test set format (28x28x3)
    '''
    # get the labels
    (_, train_labels), (_, test_labels) = tf.keras.datasets.mnist.load_data()

    # get the images
    mnistm = pk.load(open('data/mnistm_data28.pkl', 'rb')) 
    mnistm_train = mnistm['train']
    mnistm_test = mnistm['test']

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    # scale and remove pixel mean
    pixel_mean_train = np.mean(mnistm_train)
    pixel_mean_test = np.mean(mnistm_test)
    mnistm_train = (mnistm_train - pixel_mean_train) / 255
    mnistm_test = (mnistm_test - pixel_mean_test) / 255  
    return mnistm_train, mnistm_test, train_labels, test_labels

#-------------------------------------------------------------------------------------------------#
# Neural network helpers                                                                          #
#-------------------------------------------------------------------------------------------------#
def get_model(input_shape=(32, 32, 3), low_do=False):
    '''
    get a model which can be trained on mnist data set, high generalization due to high dropouts

    input_shape: input shape of the data
    low_do: specify if low or high dropout (do) layers should be used

    returns: keras model for emb-atda
    '''

    # define do level
    if low_do:
        do_val = 0.05
        noise_val = 0.075
    else:
        do_val = 0.5
        noise_val = 0.75

    # define the input into the shared network F
    input_shape = input_shape
    inp = tf.keras.layers.Input(shape=input_shape, name="main_input")
    shared_net = tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), padding="same",
                                        activation="relu", name="shared_conv0")(inp)
    shared_net = tf.keras.layers.Dropout(do_val, name="shared_do1")(shared_net)
    shared_net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="shared_maxpool0")(shared_net)

    shared_net = tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), padding="same",
                                        activation="relu", name="shared_conv1")(shared_net)
    shared_net = tf.keras.layers.Dropout(do_val, name="shared_do2")(shared_net)
    shared_net = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="shared_maxpool1")(shared_net)
    shared_net = tf.keras.layers.GaussianNoise(noise_val, name="shared_do3")(shared_net)

    shared_network_out = tf.keras.layers.Flatten(name="shared_out")(shared_net)

    # define the structure of the first network
    l1 = tf.keras.layers.Dense(48, activation="selu", name="l1_fc1")(shared_network_out)
    l1 = tf.keras.layers.AlphaDropout(do_val, name="l1_do1")(l1)

    l1 = tf.keras.layers.Dense(32, activation="selu", name="l1_fc2")(l1)
    l1 = tf.keras.layers.AlphaDropout(do_val, name="l1_do2")(l1)

    l1 = tf.keras.layers.Dense(32, activation="selu", name="l1_fc3")(l1)
    l1 = tf.keras.layers.AlphaDropout(do_val, name="l1_do4")(l1)

    pred_1 = tf.keras.layers.Dense(10, activation="softmax", name="l1_out")(l1)


    # define the structure of the second network
    lab2 = tf.keras.layers.Dense(32, activation="elu", name="l2_fc1")(shared_network_out)
    lab2 = tf.keras.layers.Dropout(do_val, name="l2_do1")(lab2)
    lab2 = tf.keras.layers.GaussianNoise(noise_val, name="l2_do2")(lab2)

    lab2 = tf.keras.layers.Dense(16, activation="elu", name="l2_fc2")(lab2)

    pred_2 = tf.keras.layers.Dense(10, activation="softmax", name="l2_out")(lab2)


    # define the structure of the third network
    l3 = tf.keras.layers.Dense(48, activation="relu", name="l3_fc1")(shared_network_out)
    l3 = tf.keras.layers.Dropout(do_val, name="l3_do0")(l3)
    l3 = tf.keras.layers.GaussianNoise(noise_val, name="l3_do1")(l3)

    l3 = tf.keras.layers.Dense(32, activation="relu", name="l3_fc2")(l3)
    l3 = tf.keras.layers.Dropout(do_val, name="l3_do2")(l3)

    pred_3 = tf.keras.layers.Dense(10, activation="softmax", name="l3_out")(l3)


    # build the net
    tri_learning_net = tf.keras.models.Model(inputs=inp, outputs=[pred_1, pred_2, pred_3])
    return tri_learning_net

def train_save_model(model, x, y, outpath, batch_size=124, lr=0.001, epc=15):
    '''
    trains a model with adam optimizer and stores its weights and model structure as a json file

    model: model to train
    x: data
    y: labels
    outpath: path to store weights and model
    batch_size: batch size for training
    lr: learning rate
    epc: epochs to train
    '''

    # define loss
    losses = {
        "l1_out": "categorical_crossentropy",
        "l2_out": "categorical_crossentropy",
        "l3_out": "categorical_crossentropy",
    }
    loss_weights = {"l1_out": 1.0, "l2_out": 1.0, "l3_out": 1.0}

    # define optimizer and compile net
    opt = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

    # train
    model.fit(x, [y, y, y], validation_split=0.1, batch_size=batch_size, epochs=epc, verbose=1)

    # save the weights
    model.save_weights(outpath + 'w.h5')

    return model

#-------------------------------------------------------------------------------------------------#
# Embedded domain adaptation helpers                                                              #
#-------------------------------------------------------------------------------------------------#
def judge_predictions(model, x_t, label_treshhold):
    '''
    uses model to predict a label of the images x_t. The model must be confident enough

    model: model with 3 outputs
    x_t: target domain data
    label_treshhold: threshhold of softmax confidence
    '''
    predicted_mnistm = []
    predicted_labels = []
    predicted_labels_idx = [False]*len(x_t)

    # get all predictions from the network
    predictions = model.predict(x_t)
    predictions_1 = predictions[0]
    predictions_2 = predictions[1]
    predictions_3 = predictions[2]

    # judge if the labelers are confident enough
    for idx in range(predictions_1.shape[0]):

        # get the classes
        pred_1 = np.argmax(predictions_1[idx, :])
        pred_2 = np.argmax(predictions_2[idx, :])
        pred_3 = np.argmax(predictions_3[idx, :])

        # check if pred are equal and high confidence
        if pred_1 == pred_2 == pred_3:
            if np.mean([np.max(predictions_1[idx, :]), np.max(predictions_2[idx, :]), np.max(predictions_3[idx, :])]) > label_treshhold:

                # add candidate to new data set
                predicted_mnistm.append(x_t[idx, :, :, :])
                predicted_labels.append(pred_1)
                predicted_labels_idx[idx] = True

    return predicted_mnistm, predicted_labels, predicted_labels_idx

def get_pseudo_labels(m, x_t, lt):
    '''
    wrapper function for judge_predictions
    '''
    # get pseudo-labels
    x, y, idx = judge_predictions(m, x_t, lt)
    idx = [i for i, x in enumerate(idx) if x]

    # return the new data set
    return np.array(x), np.array(y), np.array(idx)

#-------------------------------------------------------------------------------------------------#
# Utility                                                                                         #
#-------------------------------------------------------------------------------------------------#
def plot_grid(x, y, save, outpath):
    '''
    x: image
    y: label
    save: save image at outpath
    outpath: path of image to save
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    plt.rcParams.update({'font.size': 10})
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'figure.autolayout': True})

    size = 2*2

    # sample random images from x ,y or just first 40*8
    x_reduced = []
    y_reduced = []
    randomRows = np.random.randint(len(x), size=size)
    for i in randomRows:
        x_reduced.append(x[i, :])
        y_reduced.append(y[i])
    x_reduced = np.array(x_reduced)
    y_reduced = np.array(y_reduced)

    cf = plt.figure(1, figsize=(3.25, 3.25), dpi=65, facecolor='w', edgecolor='k')
    grid = ImageGrid(cf, 111, nrows_ncols=[2, 2], axes_pad=0.25)
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(x_reduced[i])
        grid[i].set_title(str(y_reduced[i]), fontsize=10)

    plt.show()

    if save:
        cf.savefig(outpath)

def get_class_weights(y_hot):
    '''
    calculate class weights, need one hot vector and 10 classes
    '''
    y = [y.argmax() for y in y_hot]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    class_weights = [float(i) for i in class_weights]
    return [class_weights, class_weights, class_weights]
