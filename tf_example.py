from __future__ import absolute_import, division, print_function, \
    unicode_literals

import tensorflow as tf
import numpy as np
import os
import pickle
import sys
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint

# put the text file in a good format
# let's break it up into bytes
# the final array will be a certain number of bytes
# let's say 30 bytes is one object
# so we make an array of 30-byte objects and pass it to the neural net

if len(sys.argv) != 2:
    print("Usage [tf_example]: load [0, 1]")
    exit(0)
load = int(sys.argv[1])


# custom loss function
def customLoss(yTrue, yPred):
    difference = K.round(256 * yTrue) - K.round(256 * yPred)
    nonzero = K.sum(K.cast(K.not_equal(difference, 0), tf.float32))
    return nonzero + keras.losses.mean_absolute_error(yTrue, yPred)



# K.sum, K.log...

num_epochs = 10000  # 5
bytes_per_chunk = 30
num_chunks = 3  # 10 #int(size // (8 * bytes_per_chunk))

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(30, activation='relu')
])

filepath = "/Users/blakechellew/compression/enwik8"

size = os.path.getsize(filepath) / 2  # divide in half to be safe

# initialize numpy array
all_chunks = np.zeros((num_chunks, bytes_per_chunk))

with open(filepath) as f:
    for i in range(num_chunks):
        data = f.read(8 * bytes_per_chunk)
        data_ints = np.fromstring(data, dtype=np.uint8, count=bytes_per_chunk)
        all_chunks[i] = data_ints

# verified all between 0 and 256
# data type is float64

x_train = y_train = x_test = y_test = all_chunks / 256

if not load:
    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1,
                                 save_weights_only=True)

    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=num_epochs, callbacks=[checkpoint])

    #model.save_weights('text_model_022820')

else:
    # load model
    #model.load_weights('text_model_022820')
    model.build((num_chunks, 30))
    model.load_weights('best_model.hdf5')

    y_pred = model.predict(x_test)
    y_pred = np.rint(y_pred * 256)
    y_test = np.rint(y_test * 256)

    print("y_pred", y_pred)
    print("y_test", y_test)

    plt.hist(y_pred.flatten() - y_test.flatten(), bins=50)
    plt.show()

    non_zero = np.count_nonzero(y_pred - y_test)
    total = len(y_pred.flatten())
    print("nonzero:", non_zero)
    print("total:", total)
    print("accuracy:", (total - non_zero) / total)

# model.evaluate(x_test, y_test, verbose=2)
