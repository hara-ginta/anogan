import cv2
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def load_image_and_label(pickled_files):
    # Each file contains 495 images
    IMAGE_COUNT_PER_FILE = 495
    # Image shape is 32x32x3
    ROW = 32
    COL = 32
    DIM = 3
    whole_images = np.empty((IMAGE_COUNT_PER_FILE*len(pickled_files), ROW, COL, DIM))
    whole_labels = np.empty(IMAGE_COUNT_PER_FILE*len(pickled_files))
    for i, pickled_file in enumerate(pickled_files):
        dict = _unpickle(pickled_file)
        images = dict['data'].reshape(IMAGE_COUNT_PER_FILE, DIM, ROW, COL).transpose(0, 2, 3, 1)
        whole_images[i*IMAGE_COUNT_PER_FILE:(i + 1)*IMAGE_COUNT_PER_FILE, :, :, :] = images
        labels = dict['labels']
        whole_labels[i*IMAGE_COUNT_PER_FILE:(i + 1)*IMAGE_COUNT_PER_FILE] = labels
    return (whole_images, whole_labels)

def _unpickle(pickled_file):
    import pickle

    with open(pickled_file, 'rb') as file:
        # You'll have an error without "encoding='latin1'"
        dict = pickle.load(file, encoding='latin1')
    return dict

# Function to load cucumber-9 dataset and split it into training and test data
def load_cucumber():
    (X1, y1) = load_image_and_label(['Train/data_batch_1',
                                     'Train/data_batch_2',
                                     'Train/data_batch_3',
                                     'Train/data_batch_4',
                                     'Train/data_batch_5'])
    (X2, y2) = load_image_and_label(['Test/test_batch'])
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    # 2L as normal
    normal_index = np.where(y == 0)
    # C as anomaly
    anomaly_index = np.where(y == 8)
    X_normal = X[normal_index]
    X_anomaly = X[anomaly_index]
    y_normal = y[normal_index]
    y_anomaly = y[anomaly_index]
    # split normal images into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=0.2, stratify=y_normal, random_state=0)
    X_test = np.concatenate((X_test, X_anomaly), axis=0)
    y_test = np.concatenate((y_test, y_anomaly), axis=0)
    y_test = y_test == 8
    return X_train, X_test, y_test