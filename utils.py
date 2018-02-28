import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageOps
import math
import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
import matplotlib
from collections import Counter


## This is a global vector we will use to store function pointer to our
## data augmentation Functions
aug_function = []

"""
    Init our function pointer array
"""
def init_data_augmentation():
    global aug_function
    aug_function.append(change_brightness)
    aug_function.append(flip_image)
    aug_function.append(add_random_shadow)
    return len(aug_function)

"""
    Read csv file and returned a panda Dataframe
"""
def read_data_log(filename):
    datasetLog = pd.read_csv(filename,
        names = ['center','left','right','steering','throttle','brake','speed'],
        skipinitialspace=True, header=0)
    return datasetLog

"""
    Extract data filenames from Dataframe and remove some zero samples
    that makes the data set unbalanced (and they will
    equal non zero data numerically)
    Returns numpy arrays of filenames and corresponding steering values
"""
def extract_data(data, remove_zeros=False, round_steering=True):
    if data is None:
        raise RuntimeError(" Data file contains no data")
    non_zero = data.loc[data['steering'] != 0.0]
    if remove_zeros:
        keep_values = len(non_zero)
        print("We have ", keep_values, " that are not zero")
    else:
        keep_values = -1

    zero_data = data.loc[data['steering'] == 0.0]
    filtered = remove_zero_data(zero_data, keep_values)
    x_zero = filtered['center'].values
    y_zero = filtered['steering'].values
    x_non_zero = non_zero['center'].values
    # Round steering values...we don't need all decimal digits
    if round_steering:
        y_non_zero = np.round(non_zero['steering'].values, 4)
    else:
        y_non_zero = non_zero['steering'].values
    X = np.hstack((x_zero, x_non_zero))
    Y = np.hstack((y_zero, y_non_zero))
    return X, Y

"""
    Draw histogram of the DataFrame
"""
def visualize_data(datalog):
    matplotlib.style.use('ggplot')
    datalog.plot()
    plt.show()
    # And do some statistics on steering angle
    #print(Counter(datasetLog['steering'].values))
    labels, values = zip(*sorted(Counter(datalog['steering'].values).items()))
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels, rotation='vertical')
    plt.show()


"""
    Get rid of zero steering data values
    if how_many is set to -1 all the data are kept
    otherwise number of data equal to how_many data will be kept
"""
def remove_zero_data(zero_data, how_many):
    if zero_data is None:
        return None
    print("Zero data info: ")
    zero_data.info()
    num_samples = len(zero_data)
    print("We have ", num_samples, "with zero value")
    # if -1 is passed then keep all
    if how_many == -1:
        how_many = num_samples
    keep_fraction = how_many/num_samples
    print("We will keep ", keep_fraction * 100, "% of the zero data")
    zero_data_shuffled = zero_data.sample(frac=keep_fraction, replace=False)
    print("Zero data info shuffled: ")
    zero_data_shuffled.info()
    return zero_data_shuffled



"""
    Read files and convert to RGB
    Given file name returns a numpy array with images one per row
"""
def read_images(file_names):
    images = []
    for f in file_names:
        img = cv2.imread(f)
        # cv2.imshow("sample" ,img)
        # cv2.waitKey(0)
        images.append(img)
    return np.array(images)

"""
    This will resize images to be passed to CNN.
    Images will be cropped too between 2/5 and 4/5 for the heigh size and
    2/10 and 9/10 for the width (those are the portion of images that will be
    kept.
"""
def preprocess_images(images, swap_color=False):
    preprocessed = []
    for i in range(images.shape[0]):
        tmp = images[i]
        if swap_color:
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

        crop_height1 = math.ceil(tmp.shape[0] / 7) * 2
        crop_height2 = math.ceil(tmp.shape[0] / 10) * 8
        crop_width1 = math.ceil(tmp.shape[1] / 10) * 1
        crop_width2 = math.ceil(tmp.shape[1] / 10) * 9
        tmp = tmp[crop_height1:crop_height2, crop_width1:crop_width2]
        tmp = cv2.resize(tmp, (100,33))
        # let's blur the image...
        tmp = cv2.GaussianBlur(tmp,(5,5),0)
        preprocessed.append(tmp)
    data = np.array(preprocessed)
    return data


def augment_dataset_single(files, steering):
    X_aug = []
    Y_aug = []
    # how many transformation we have in our menu....
    how_many = len(aug_function)
    for i in range(files.shape[0]):
        # We have the lenght of the aug_function vector + 3 (two other function)
        # as the randint(0, a) returns an integer between [0, a ) [0...4)
        fate = np.random.randint(0, how_many + 3)
        img = read_images([files[i]])
        img = img[0]
        if (fate < 3):
            # Brightness or flip
            x_tmp, y_tmp = aug_function[fate](img, steering[i])
            X_aug.append(x_tmp)
            Y_aug.append(y_tmp)
        elif fate == 3:
            # Use left and right images to recover from the side and choose one of the two
            left_right, left_right_steering = use_left_right_images(files[i], steering[i])
            which_one = np.random.randint(0, 2)
            x_tmp = left_right[which_one]
            y_tmp = left_right_steering[which_one]
            # Change brightness here
            x_tmp, y_tmp = aug_function[0](x_tmp, y_tmp)
            X_aug.append(x_tmp)
            Y_aug.append(y_tmp)
        else:
            # Return the untouched image
            X_aug.append(img)
            Y_aug.append(steering[i])
    return np.array(X_aug), np.array(Y_aug)

"""
    Change the brigtness to the image and return the new image.
    Second parameter is unused :)
"""
def change_brightness(img, y):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .2 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_bright
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image, y


"""
 Create a horizontally flipped copy of the images passed in filenames
 Steering angle will be multiplied by -1 in order to reverse it
"""
def flip_image(img, steering):
    flipped_image = np.fliplr(img)
    flipped_stering = -1 * steering
    return flipped_image, flipped_stering


def use_left_right_images(filename, steering):
    result_images = []
    result_steering = []
    #left_steering_correction = 0.27
    #right_steering_correction = -0.27
    left_steering_correction = 0.32
    right_steering_correction = -0.32
    left_filename = filename.replace('center', 'left')
    right_filename = filename.replace('center', 'right')
    img_left = read_images([left_filename])
    if img_left[0] is None:
        print("File... ", left_filename, " Does not exists")
    left_steering = steering + left_steering_correction
    result_images.append(img_left[0])
    result_steering.append(left_steering)
    img_right = read_images([right_filename])
    if img_right[0] is None:
        print("File... ", left_filename, " Does not exists")
    right_steering = steering + right_steering_correction
    result_images.append(img_right[0])
    result_steering.append(right_steering)
    return result_images, result_steering

'''
 This function has been taken from
 https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.74m9mw2e9
 with very very little changes
 Second parameter is unused :)
'''
def add_random_shadow(image, steering):
    top_y = image.shape[1]*np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1]*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = np.zeros((160,320))
    grid = np.mgrid[0:image.shape[0],0:image.shape[1]]
    X_m = grid[0]
    Y_m = grid[1]
    shadow_mask[((X_m-top_x)* (bot_y-top_y) - (bot_x - top_x) * (Y_m-top_y) >= 0)] = 1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image, steering
