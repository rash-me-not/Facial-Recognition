from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KNNManual:

    def __init__(self):
        pass

    def train(self, train_features, train_labels):
        """Save the training data for validation/predictions"""

        # If the train features is None, then use the x_train from the original dataset,
        # else train on the given features and labels
        self.x_train = train_features
        self.y_train = train_labels

    def train_wd_cross_validation(self, x_train, y_train, num_folds, k_choices, dist_metric):
        """
        Perform k-fold cross validation to find the best value of k. For each k, run the KNN method multiple times.
        Here based on the number of folds (n), we train the model using n-1 proportion of dataset, and validate on one
        portion of data. This process is repeated n times for k different values
        Store the accuracies for all folds and all k values.
        Return the model with the maximum accuracy
        """
        # split up the training data and labels into folds.
        x_train_folds = np.array_split(x_train, num_folds)
        y_train_folds = np.array_split(y_train, num_folds)

        k_to_accuracies = {}  # k_to_accuracies[k]: a list giving the different accuracy values
        acc_k = np.zeros((len(k_choices), num_folds), dtype=np.float)
        # call classifiers multiple times to generate the cross-validation
        for ik, k in enumerate(k_choices):
            for i in range(num_folds):
                train_set = np.concatenate((x_train_folds[:i] + x_train_folds[i + 1:]))
                label_set = np.concatenate((y_train_folds[:i] + y_train_folds[i + 1:]))
                self.train(train_set, label_set)
                y_pred_fold = self.predict(x_train_folds[i], dist_metric,  k=k)
                acc_k[ik, i] = self.get_accuracy(y_pred_fold, y_train_folds[i])
            print("Accuracy for k: %d processed"%k)
            k_to_accuracies[k] = acc_k[ik]

        # Print out the computed accuracies
        for k in sorted(k_to_accuracies):
            for accuracy in k_to_accuracies[k]:
                print('k = %d, accuracy = %f' % (k, accuracy))

        # plot the raw observations
        for k in k_choices:
            accuracies = k_to_accuracies[k]
            plt.scatter([k] * len(accuracies), accuracies)

        # plot the trend line with error bars that correspond to standard deviation
        accuracies_mean = {}
        accuracies_std = {}
        for k, v in sorted(k_to_accuracies.items()):
            accuracies_mean[k] = np.mean(v)
            accuracies_std[k] = np.mean(v)

        plt.errorbar(k_choices, accuracies_mean.values(), yerr=accuracies_std)
        plt.title('Cross-validation on k')
        plt.xlabel('k')
        plt.ylabel('Cross-validation accuracy')
        plt.show()
        return max(accuracies_mean, key=accuracies_mean.get)

    def predict(self, test_features, dist_metric, k):
        """ Returns the predicted value for the test-features based on distance metric and k"""

        # If the test_features is None, then use the x_test from the original dataset to predict
        # (This happens when we are not using the cross-validation).
        if dist_metric == "Manhattan":
            dists = self.compute_distances_manhattan(test_features)
        elif dist_metric == "Euclidean":
            dists = self.compute_distances_euclidean(test_features)
        else:
            raise ValueError('Invalid value %d for distance metric' % dist_metric)
        return self.predict_labels(dists, k=k)

    def compute_distances_manhattan(self, X):
        num_test = X.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train), dtype=self.y_train.dtype)

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            dists[i,:] = np.sum(np.abs(self.x_train - X[i, :]), axis=1)
        return dists

    def compute_distances_euclidean(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        """
        num_test = X.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train), dtype=self.y_train.dtype)
        for i in range(num_test):  #
            # Compute the l2 distance between the ith test point and all training  points, and store the result in
            # dists[i, :].
            dists[i, :] = np.sqrt(np.sum(np.square(self.x_train - X[i,:]), axis = 1))
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to the ith test point.
            # Use the distance matrix to find the k nearest neighbors of the ith testing point,
            # and use self.y_train to find the labels of these neighbors. Store these labels in closest_y.

            top_k_indx = np.argsort(dists[i])[:k]
            closest_y = self.y_train[top_k_indx]

            # Store this label in y_pred[i]. Break ties by choosing the smaller label.
            vote = Counter(closest_y)
            count = vote.most_common()
            y_pred[i] = count[0][0]

        return y_pred

    def get_accuracy(self, y_pred, y_test):
        num_test = y_pred.shape[0]
        num_correct = np.sum(y_pred == y_test)
        accuracy = float(num_correct) / num_test
        return accuracy
