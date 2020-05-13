import itertools
import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn_sklearn import KNNSklearn
from sklearn.metrics import confusion_matrix
from cnn_tensorflow import CNNTensorflow
from linearclassifier import LinearClassifier
from knn_manual import KNNManual
from hog import Hog_descriptor
import pickle
import cv2


class Main:
    def __init__(self, file):
        self.file = file

    def generate_dataset(self, read_from_pickle = False):
        """Read the source file and return a dictionary with x_train, y_train, x_val, y_yal, x_test and y_test"""

        data_dict = {}
        data_path = os.path.join(os.path.dirname(__file__), 'data')

        if not read_from_pickle:
            df = pd.read_csv(file, header=0)
            if not os.path.exists(data_path):
                os.mkdir(data_path)

            train_samples = df[df.Usage == "Training"]
            val_samples = df[df.Usage == "PrivateTest"]
            test_samples = df[df.Usage == "PublicTest"]

            samples_dict = {"train": train_samples, "val": val_samples, "test": test_samples}
            for type, samples in samples_dict.items():
                data_dict["x_" + type],data_dict["x_feat_" + type], data_dict["y_" + type] = self.preprocess(samples)
            pickle.dump(data_dict, open(os.path.join(data_path, "data.p"), "wb"))
        else:
            data_dict = pickle.load(open(os.path.join(data_path, "data.p"), "rb"))

        self.visualize(data_dict["x_train"], data_dict["y_train"])
        return data_dict["x_train"], data_dict["x_feat_train"], data_dict["y_train"], \
               data_dict["x_val"], data_dict["x_feat_val"], data_dict["y_val"], \
               data_dict["x_test"], data_dict["x_feat_test"], data_dict["y_test"]

    def visualize(self, x, y):
        """Visualize some examples from the dataset"""

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        # Take a look at the first 25 images of the dataset
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            image = x[i].reshape(48, 48)
            plt.imshow(image, cmap='gray')
            plt.xlabel(emotion_labels[y[i]])
        plt.show()

    def preprocess(self, samples):
        """format the features and labels as necessary for processing"""

        y = []
        x = []
        x_feat = []
        for idx, image in samples.iterrows():
            y.append(int(image.emotion))
            image_pixel = np.asarray([float(pix) for pix in image.pixels.split(" ")])
            x.append(image_pixel)  # normalizing
            hog = Hog_descriptor(image_pixel, cell_size=2, bin_size=8)
            vector, image = hog.extract()
            x_feat.append(vector)
        return np.asarray(x), np.asarray(x_feat), np.asarray(y)

    def gaussian_kernel(self, size, sigma=1, verbose=False):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = self.dnorm(kernel_1D[i], 0, sigma)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
        kernel_2D *= 1.0 / kernel_2D.max()
        if verbose:
            plt.imshow(kernel_2D, interpolation='none', cmap='gray')
            plt.title("Image")
            plt.show()
        return kernel_2D

    def dnorm(self, x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

    def convolution(self, image, kernel, average=False, verbose=False):
        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape
        output = np.zeros(image.shape)
        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)
        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
        if average:
            output[row, col] /= kernel.shape[0] * kernel.shape[1]

        print("Output Image size : {}".format(output.shape))

        if verbose:
            plt.imshow(output, cmap='gray')
            plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
            plt.show()

        return output

    def gaussian_blur(self, image, kernel_size, verbose=False):
        kernel = self.gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
        return self.convolution(image, kernel, average=True, verbose=verbose)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):

        # Add Normalization Option
        '''prints pretty confusion metric with normalization option '''
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color ="white" if cm[
                i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


if __name__ == "__main__":
    file = "../../fer2013/fer2013.csv"
    main = Main(file)

    x_train, x_feat_train, y_train, \
    x_val, x_feat_val, y_val, x_test, x_feat_test, y_test = main.generate_dataset(read_from_pickle = True)
    cnn = CNNTensorflow()

    """Training and Validation"""
    model = cnn.train_and_validate(x_train, x_feat_train, y_train,
                                   x_val, x_feat_val, y_val, scale=False)

    x_test_reshaped = cnn.reshape_features(x_test, scale=False)
    y_test_categ = cnn.to_categ(y_test, num_classes=7)

    """Prediction"""
    y_pred_categ = cnn.predict(model, x_test_reshaped, x_feat_test)

    """Classification metrics"""
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    classification_metrics = cnn.get_classification_metrics(y_pred_categ.argmax(axis=1), y_test_categ.argmax(axis=1), emotion_labels)
    print(classification_metrics)

    confusion_matrix = confusion_matrix(y_test_categ.argmax(axis=1), y_pred_categ.argmax(axis=1))
    main.plot_confusion_matrix(confusion_matrix, emotion_labels)


    # knn = KNNSklearn(data)
    # lc = LinearClassifier(data)
    #
    # knn_man = KNNManual()

    # k_list = [1, 2, 5]
    # num_folds = 3
    # k = knn.train_wd_cross_validation(x_train, y_train, num_folds, k_list, "Manhattan")
    # k = knn_man.train_wd_cross_validation(x_feat_train, y_train, num_folds, k_list, "Manhattan")

    # print("Best k: %d" % k)

    # Retrain the model with the best k and predict on the test data
    # knn.train(x_train, y_train)
    # knn_man.train(x_feat_train, y_train)
    # y_pred = knn_man.predict(x_feat_test, "Manhattan", k)
    # accuracy = knn_man.get_accuracy(y_pred, y_test)
    # print('Final Result:=> accuracy: %f' % (accuracy))
    #
    # # k_list = [1,2]
    # # knn.train_and_validate(k_list, "Manhattan")
    # # lc.train_and_validate()