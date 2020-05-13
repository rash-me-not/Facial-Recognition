import datetime

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D, Dense, concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Model
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2


class CNNTensorflow(object):

    def __init__(self):
        pass

    def get_cnn_model(self, x, droprate=0.2):
        """Creates a CNN model where the last layer has 7 nodes"""
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(x.shape[1], x.shape[2], 1)))
        model.add(BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(droprate))

        model.add(layers.Dense(7, activation='softmax'))
        return model

    def get_combined_model(self, img_train, img_train_feat, droprate=0.2):
        inp_img = Input(shape=(img_train.shape[1], img_train.shape[2], 1))
        inp_hog = Input(shape=(img_train_feat.shape[1],))

        img = Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inp_img)
        img = BatchNormalization()(img)
        img = MaxPooling2D(pool_size=(1, 5))(img)

        img = Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(img)
        img = BatchNormalization()(img)
        img = MaxPooling2D(pool_size=(1, 2))(img)

        img = Conv2D(16, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(img)
        img = BatchNormalization()(img)
        img = MaxPooling2D(pool_size=(1, 2))(img)

        img = Flatten()(img)
        img = Dense(16, activation="relu")(img)
        img = BatchNormalization()(img)
        img = Dropout(droprate)(img)
        # apply another FC layer, this one to match the number of nodes
        # coming out of the HOG
        img = Dense(4, activation="relu")(img)
        model_img = Model(inputs=inp_img, outputs=img)

        # the second branch opreates on the second input
        hog = Dense(32, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inp_hog)
        hog = Dense(16, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(hog)
        hog = Dense(4, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(hog)
        model_hog = Model(inputs=inp_hog, outputs=hog)

        combined = concatenate([model_img.output, model_hog.output])
        result = Dense(4, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(combined)
        result = Dense(7, activation='softmax')(result)

        model = Model(inputs=[model_img.input, model_hog.input], outputs=result)
        return model

    def reshape_features(self, x, scale=False):
        x_reshaped = x.reshape(x.shape[0], 48, 48)
        if scale:
            '''Scaling to 96*96 using up pyramid function'''
            images = []
            for image in x_reshaped:
                images.append(cv2.pyrUp(image))
            x_reshaped = np.asarray(images)
        x_reshaped = np.expand_dims(x_reshaped, 3)
        return x_reshaped

    def to_categ(self, y, num_classes):
        return to_categorical(y, num_classes=num_classes)

    def train_and_validate(self, x_train, x_feat_train, y_train, x_val, x_feat_val, y_val, batch_size=32, with_hog=False, dist_metric="accuracy", scale=False):
        start = datetime.datetime.now()
        x_train_reshaped = self.reshape_features(x_train, scale)
        x_val_reshaped = self.reshape_features(x_val, scale)

        y_train_categ = self.to_categ(y_train, num_classes=7)
        y_val_categ = self.to_categ(y_val, num_classes=7)

        if(with_hog):
            model = self.get_combined_model(x_train_reshaped, x_feat_train, droprate=0.5)
        else:
            model = self.get_cnn_model(x_train_reshaped, droprate=0.5)
        print(model.summary())
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=[dist_metric])

        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=0)
        reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=0,
                                           mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)

        history = model.fit([x_train_reshaped,x_feat_train], y_train_categ,
                            epochs=15,
                            batch_size=batch_size,
                            validation_data=([x_val_reshaped,x_feat_val], y_val_categ),
                            shuffle=True)

        (eval_loss, eval_accuracy) = model.evaluate(
            [x_val_reshaped,x_feat_val], y_val_categ, batch_size=batch_size, verbose=0)

        print("[INFO] accuracy: {:.2f} % ".format(eval_accuracy * 100))
        print("[INFO] Loss: {}".format(eval_loss))
        end = datetime.datetime.now()
        elapsed = end - start
        print('Time: ', elapsed)
        self.plot_acc_loss(history)
        return model

    def plot_acc_loss(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'r', label ='Training acc')
        plt.plot(epochs, val_acc, 'b', label ='Validation acc')
        plt.title('Training and validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'r', label ='Training loss')
        plt.plot(epochs, val_loss, 'b', label ='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    def predict(self, model, x_test, x_feat_test):
        return model.predict([x_test, x_feat_test])

    def get_classification_metrics(self, y_pred, y_test, emotion_labels):
        classification_metrics = metrics.classification_report(y_test, y_pred, target_names=emotion_labels)
        return classification_metrics
