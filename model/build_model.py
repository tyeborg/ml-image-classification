# Import the appropriate libraries.
import numpy as np
import os
import argparse
import cv2 as cv
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array

# Import MyModel class.
from model import MyModel

def data_extractor(path,height=32,width=32):
    data=[]
    labels = []
    imagepaths = list(paths.list_images(path))
    for imagepath in imagepaths:
        # Read the image.
        image = cv.imread(imagepath)
        # Scale the image to the ideal size and dimension (preprocessing).
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image,(height,width),interpolation=cv.INTER_AREA)
        # Convert the image to an array.
        image = img_to_array(image)

        # Determine the label of the current image.
        label = imagepath.split(os.sep)[-2]
        label = int(label)
        labels.append(label)
        data.append(image)

    # Return a the image data along with each images' label. 
    return np.array(data,dtype='float')/255.0,np.array(labels)

def build():
    # Initialize the paths of the training and testing data.
    train_folder = '../data/train_folder/'
    test_folder = '../data/test_folder/'

    # Split the data into training and testing sets.
    x_train, y_train = data_extractor(train_folder)
    x_test, y_test = data_extractor(test_folder)

    # Training will consist of 70% of the data.
    #print(f'Train Amount: {len(x_train)}')
    # Testing will consist of 30% of the data.
    #print(f'Test Amount: {len(x_test)}')

    # Reshape training and test set Y to have only 1 column.
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    model = MyModel(x_train, y_train, x_test, y_test)
    model.build()

if __name__ == '__main__':
    try:
        build()
    except Exception:
        print("\nSomething has gone terribly wrong...")