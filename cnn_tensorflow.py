import tensorflow as tf
from tensorflow import keras
import numpy as np

class CNNTensorflow(object):

    def __init__(self, data):
        self.data = data
        self.x_train = self.data["x_train"]
        self.y_train = self.data["y_train"]
        self.x_val = self.data["x_val"]
        self.y_val = self.data["y_val"]
        self.x_test = self.data["x_test"]
        self.y_test = self.data["x_test"]

    def get_cnn_model(self):
        """Creates a CNN model where the last layer has 7 nodes"""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(7)
            ])
        
        return model

    def train_and_validate(self, dist_metric="accuracy"):
        model = self.get_cnn_model()
        # Configure the model for training: sets the loss function, optimizer, and metrics
        model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[dist_metric])
        model.fit(self.x_train, self.y_train, epochs=7)
        
        loss_model, acc_model = model.evaluate(self.x_train, self.y_train)

        # Print out the computed accuracy
        print('loss = %f, accuracy = %f' % (loss_model, acc_model))

        return acc_model

    def predict(self):
        pass