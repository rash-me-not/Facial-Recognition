import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import models, layers


class CNNTensorflow(object):

    def __init__(self):
        pass

    def get_cnn_model(self, x):
        """Creates a CNN model where the last layer has 7 nodes"""
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(7, activation='softmax'))
        return model

    def train_and_validate(self, x_train, y_train, dist_metric="accuracy"):
        model = self.get_cnn_model()
        # Configure the model for training: sets the loss function, optimizer, and metrics
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[dist_metric])
        model.fit(x_train, y_train, epochs=7)

        loss_model, acc_model = model.evaluate(x_train, y_train)

        # Print out the computed accuracy
        print('loss = %f, accuracy = %f' % (loss_model, acc_model))

        return acc_model

    def predict(self):
        pass
