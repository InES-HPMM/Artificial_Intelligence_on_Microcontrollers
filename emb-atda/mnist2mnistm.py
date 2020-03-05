"""
Prototype of the embedded asymmetric tri-training for unsupervised domain adaptation (emb-atda)[1],
inspired by [2].

Adapts a neural net trained originally trained on on mnist data to mnistm data. Algorithm is
adjusted from the original algorithm that it could be ported to an embedded target:
    - reduced network size
    - no source domain data for training
    - implemented in plain keras

:Author: Raphael Zingg zing@zhaw.ch
:Copyright: 2020 ZHAW / Institute of Embedded Systems

:References:
[1] Raphael Zingg, Matthias Rosenthal, "Artificial Intelligence on Microcontrollers",
    Embedded World Conference Proceeding 2020
[2] S. Kuniaki, U. Yoshitaka and Harada Tatsuya, ”Asymmetric tri-training
    for unsupervised domain adaptation”, Proceedings of the 34th Inter-
    national Conference on Machine Learning, Volume 70, pp 2988-2997, 2017
"""
import sys
import numpy as np
import tensorflow as tf
from utils import return_mnist, return_mnistm, get_model, \
                  train_save_model, plot_grid, get_pseudo_labels, get_class_weights


# --------------------------------------------------------------------------------------------------
# Settings / Hyperparameters
# --------------------------------------------------------------------------------------------------
TRAIN_FRESH = str(sys.argv[1]) # train model fresh on source or reload weights from 'outpath'
ENABLE_PLOT = str(sys.argv[2]) # plot images
N_BOOST = 500                  # number of data/labels pairs used for boost
LABEL_TRESHOLD = 0.98          # label threshhold
LR_SOURCE = 0.0001             # source learning rate
LR_TARGET = 0.0001             # target learning rate
EPOCH_SOURCE = 15              # source training epochs
DOMAIN_ADAPTATION_STEPS = 30   # domain adaptation steps
OUTPUT_FOLDER = 'output/'      # location of the output folder

# --------------------------------------------------------------------------------------------------
# Load the data MNIST (X_S) and MMINSTM (x_t)
# --------------------------------------------------------------------------------------------------
X_S, _, Y_S, _ = return_mnist()
X_T, X_T_VAL, Y_T, Y_T_VAL = return_mnistm()

if ENABLE_PLOT == 'Plot':
    plot_grid(X_S, [np.argmax(i) for i in Y_S], False, '')
    plot_grid(X_T, [np.argmax(i) for i in Y_T], False, '')

# --------------------------------------------------------------------------------------------------
# Train on source domain data (X_S) and store the network or get trained
# --------------------------------------------------------------------------------------------------
if TRAIN_FRESH == 'Train':
    model = get_model(input_shape=(X_S.shape[1], X_S.shape[1], 3), low_do=False)
    train_save_model(model, X_S, Y_S, OUTPUT_FOLDER, batch_size=124, lr=LR_SOURCE, epc=EPOCH_SOURCE)

# --------------------------------------------------------------------------------------------------
# Prepare emb-atda algorithm
# --------------------------------------------------------------------------------------------------

# load the model for domain adaptation, which has the trained weights but lower dropout rates
trained_net = get_model(input_shape=(X_S.shape[1], X_S.shape[1], 3), low_do=True)
trained_net.load_weights(OUTPUT_FOLDER + 'w.h5')

# adjust loss and lr for emb-atda
losses = {"l1_out": "categorical_crossentropy",
          "l2_out": "categorical_crossentropy",
          "l3_out": "categorical_crossentropy", }
loss_weights = {"l1_out": 0.5, "l2_out": 0.5, "l3_out": 0.25}
trained_net.compile(optimizer=tf.keras.optimizers.Adam(LR_TARGET), loss=losses,
                    loss_weights=loss_weights, metrics=['accuracy'])

# evaluate source only trained net on target domain test set
res_t = trained_net.evaluate(X_T_VAL, [Y_T_VAL, Y_T_VAL, Y_T_VAL], verbose=0)
print('\n\nAccuracy of source only trained network on target domain test data:', res_t[-1], '\n\n')

# use random labels from x_t and X_S to boost
random_rows = np.random.randint(len(X_T), size=N_BOOST)
x_t_b = X_T[random_rows]
y_t_b = Y_T[random_rows]

# --------------------------------------------------------------------------------------------------
# Run emb-atda, this could be done on an embedded target
# --------------------------------------------------------------------------------------------------
for step in range(1, DOMAIN_ADAPTATION_STEPS):

    # get pseudo labels
    x_t_p, y_t_p, idx = get_pseudo_labels(trained_net, X_T, LABEL_TRESHOLD)
    n = max(int((step + 1) / 20 * len(X_T)), 1000)

    # create mixed dataset
    if n > len(x_t_p):
        n = len(x_t_p)
    random_rows = np.random.randint(len(x_t_p), size=n)
    x_t_p_sampled = x_t_p[random_rows]
    y_t_p_sampled = y_t_p[random_rows]
    y_t_p_sampled = tf.keras.utils.to_categorical(y_t_p_sampled, num_classes=10)

    x_t_c = np.array(np.vstack([x_t_b, x_t_p_sampled]))
    y_t_c = np.array(np.vstack([y_t_b, y_t_p_sampled]))

    if ENABLE_PLOT == 'Plot':
        plot_grid(x_t_p, y_t_p, False, '')

    # train labelers alternating
    if (step % 2) == 0:
        for layer in trained_net.layers:
            if 'l1' in layer.name:
                layer.trainable = True
            if 'l2' in layer.name:
                layer.trainable = False
            if 'l3' in layer.name:
                layer.trainable = True
    else:
        for layer in trained_net.layers:
            if 'l1' in layer.name:
                layer.trainable = False
            if 'l2' in layer.name:
                layer.trainable = True
            if 'l3' in layer.name:
                layer.trainable = True
    class_weights = get_class_weights(y_t_c)
    trained_net.compile(optimizer=tf.keras.optimizers.Adam(LR_TARGET), loss=losses,
                        loss_weights=loss_weights, metrics=['accuracy'])
    trained_net.fit(x_t_c, [y_t_c, y_t_c, y_t_c], batch_size=64, epochs=1,
                    verbose=0, class_weight=class_weights)

    # evalulation
    res_t = trained_net.evaluate(X_T_VAL, [Y_T_VAL, Y_T_VAL, Y_T_VAL], verbose=0)
    print('\nStep:', step, 'Acc on target domain test data:', res_t[-1], 'Len y_t_p:', len(y_t_p), '\n')

# save the domain adapted weights
trained_net.save_weights(OUTPUT_FOLDER + 'da_w.h5')
