import numpy as np
import os
import argparse
from imutils import paths
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer,Conv2D,Activation,MaxPool2D,Dense,Flatten,Dropout

class MyModel():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def build(self):
        # Declare the input shape parameters of an input image.
        height = 32
        width = 32
        depth = 1
        input_shape = (width, height, depth)

        # Set up a Sequential model.
        model = Sequential()

        # Specify the shape of the Convolutional kernel: (3, 3)
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        # Apply Dropout to increase efficiency and eliminate useless nodes.
        model.add(Dropout(0.5))
        # Prediction of the next value.
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        # Compile and fit the model.
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=15, batch_size=32)

        model.save('smile-model.h5')

        # Plot the model performance and save figure into folder.
        self.plot_performance(model_history)

    def plot_performance(self, model_fit):
        acc = model_fit.history['accuracy']
        val_acc = model_fit.history['val_accuracy']
        loss_ = model_fit.history['loss']
        val_loss_ = model_fit.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.xlim([0, 15])
        plt.ylim([0.8, 1])

        plt.legend(loc='upper left')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss_, label='Training Loss')
        plt.plot(val_loss_, label='Validation Loss')
        plt.xlim([0, 15])
        plt.ylim([0, 1.0])
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.savefig('../model_eval/epoch-performance.png', bbox_inches='tight')