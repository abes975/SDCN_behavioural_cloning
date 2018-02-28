import os
import argparse
import json
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D
from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
import cv2
import math

import utils

def generate(files, steering, batch_size, augment_data=True):
    #data = shuffle(data)
    num_examples = len(files)
    offset = num_examples
    i = 1
    while True:
        if (offset + batch_size) >= num_examples:
            offset = 0
            i = 1
            files_s, steering_s = shuffle(files, steering)
        for offset in range(0, num_examples, batch_size):
            i += 1
            end = offset + batch_size
            if end >= num_examples:
                end = num_examples
            filename_x, batch_y = files_s[offset:end], steering_s[offset:end]
            if augment_data:
                batch_x, batch_y = utils.augment_dataset_single(filename_x, batch_y)
            else:
                batch_x = utils.read_images(filename_x)
            # Rescale and resize only
            batch_x = utils.preprocess_images(batch_x, False)

            yield batch_x.astype('float32'), batch_y.astype('float32')


def model():
  ch, row, col = 3, 33, 100  # camera format
  model = Sequential()
  model.add(Lambda(lambda x: x/255.-0.5,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
  model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode="valid"))
  model.add(ELU())
  model.add(Dropout(.7))
  model.add(Convolution2D(32, 5, 5, subsample=(1, 1), border_mode="valid"))
  model.add(ELU())
  model.add(Dropout(.5))
  model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="valid"))
  model.add(ELU())
  model.add(Dropout(.5))
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
  model.add(ELU())
  model.add(Dropout(.5))
  model.add(Flatten())
  model.add(Dense(48))
  model.add(ELU())
  model.add(Dropout(.5))
  model.add(Dense(32))
  model.add(ELU())
  model.add(Dropout(.5))
  model.add(Dense(16))
  model.add(ELU())
  model.add(Dense(1))
  model.compile(optimizer="adam", loss="mse", lr=0.0001)

  return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--data_filename', type=str, default='driving_log.csv', help='Data file name.')
    parser.add_argument('--model', type=str, default='./model', help='Output file name.')
    parser.add_argument('--visualize', type=bool, default=False, help='Visualize data distribution.')

    args = parser.parse_args()
    outfile = args.model

    data_filename = args.data_filename
    dirname = os.path.dirname(data_filename)
    print("Data in dir: ", dirname, " filename: ", os.path.basename(data_filename))

    # initialize our array of function pointer that will be used in transformation:
    aug_func_count = utils.init_data_augmentation()
    # We create 4 images for every input sample
    # change brigtness
    # flip
    # use left and right camera image
    # add_random_shadow
    # the image itself
    # So we use this factor as num sample sizes for training,
    # aug_factor = 4
    aug_factor = 5

    print("We have ", aug_func_count, " augmentation function in our model"
        " with an augmentation factor of ", aug_factor)

    # Read our dataset
    dataLog_orig = utils.read_data_log(data_filename)
    # Get rid of some noisy data...
    utils.visualize_data(dataLog_orig)

    dataLog = dataLog_orig.loc[dataLog_orig['throttle'] > 0.25 ]
    print("Loaded data info: ")
    dataLog.info()

    filenames, steering = utils.extract_data(dataLog, remove_zeros=False, round_steering=True)

    total_sample = len(filenames)
    train_files, val_files, train_steering, val_steering = train_test_split(filenames, steering, test_size=0.33, random_state=543)
    train_samples = len(train_files)
    val_samples = len(val_files)
    print("Total Sample: ", total_sample, " Training samples : ", train_samples, " Validation samples: ", val_samples)
    batch_size = args.batch
    epochs = args.epochs

    model = model()
    model.summary()

    # Do some preparation in order to save our model every epoch..and then test
    # Different epochs :) on the simulator...
    ##### weight_file = outfile + ".h5"
    json_file = outfile + ".json"

    model_json = model.to_json()
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(json_file, "w") as js_file:
        js_file.write(model_json)
    print("Saved model: ", json_file)


    # Create filename dinamically appending epoch number
    filepath = outfile + "-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                    save_best_only=False, save_weights_only=True, mode='auto')
    callbacks_list = [checkpoint]

    history = model.fit_generator(
        generate(train_files, train_steering, batch_size, True),
        samples_per_epoch = train_samples * aug_factor,
        nb_epoch = epochs,
        validation_data = generate(val_files, val_steering, batch_size, False),
        nb_val_samples = val_samples,
        callbacks=callbacks_list
    )

    plt.figure(figsize=(6, 3))
    plt.plot(history.history['loss'], label="training loss")
    plt.plot(history.history['val_loss'], label="validation loss")
    plt.legend(loc=2, borderaxespad=0.)
    plt.ylabel('error')
    plt.xlabel('iteration')
    plt.title('training error')
    plt.show()
    # The last 2 lines are useufl to prevent a tf bug
    from keras import backend as K
    K.clear_session()
