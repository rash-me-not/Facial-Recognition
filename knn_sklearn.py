from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import matplotlib.pyplot as plt


class KNNSklearn:

    def __init__(self):
        pass

    def get_knn_model(self, k, dist_metric):
        """When p = 1, use manhattan_distance , else euclidean_distance for p = 2"""

        dist_dict = {"Manhattan": 1, "Euclidean": 2}
        model = KNeighborsClassifier(n_neighbors=k, p=dist_dict[dist_metric])
        return model

    def train_and_validate(self, x_train, y_train, x_val, y_val, k_list, dist_metric):
        """ Iterates through different values of k (number of nearest neighbors), fetches a knn classifier based on the
        distance metric (Manhattan/Euclidean), trains the model and determines the accuracy on the validation data.
        Returns the model with the highest validation accuracy """

        validation_accuracies = []
        for k in k_list:
            model = self.get_knn_model(k, dist_metric)
            model.fit(x_train, y_train)
            filename = 'finalized_model_' + str(k) + '_neighbor.sav'
            pickle.dump(model, open(filename, 'wb'))
            # model = pickle.load(open(filename, 'rb'))
            acc = model.score(x_val, y_val)
            # keep track of what works on the validation set
            validation_accuracies.append((k, acc, model))
        print(validation_accuracies)

        # Find model with highest validation accuracy and return the model
        max_acc = 0
        for acc in validation_accuracies:
            if acc[1] > max_acc:
                max_acc = acc[1]
                acc_model = acc[2]
        return acc_model

    def train_wd_cross_validation(self, x_train, y_train, num_folds, k_choices, dist_metric):
        """
        Perform k-fold cross validation to find the best value of k. For each k, run the KNN method multiple times.
        Here based on the number of folds (n), we train the model using n-1 proportion of dataset, and validate on one
        portion of data. This process is repeated n times for k different values
        Store the accuracies for all folds and all k values.
        Return the model with the maximum accuracy
        """
        # split up the training data and labels into folds.
        X_train_folds = np.array_split(x_train, num_folds)
        y_train_folds = np.array_split(y_train, num_folds)

        k_to_accuracies = {}  # k_to_accuracies[k]: a list giving the different accuracy values
        acc_k = np.zeros((len(k_choices), num_folds), dtype=np.float)
        max_acc = 0

        # call classifiers multiple times to generate the cross-validation
        for ik, k in enumerate(k_choices):
            for i in range(num_folds):
                train_set = np.concatenate((X_train_folds[:i] + X_train_folds[i + 1:]))
                label_set = np.concatenate((y_train_folds[:i] + y_train_folds[i + 1:]))
                model = self.get_knn_model(k, dist_metric)
                model.fit(train_set, label_set)
                acc = model.score(X_train_folds[i], y_train_folds[i])
                acc_k[ik, i] = acc
                if acc > max_acc:
                    max_acc = acc
                    best_model = model
                k_to_accuracies[k] = acc_k[ik]

        # Print out the computed accuracies
        for k in sorted(k_to_accuracies):
            for accuracy in k_to_accuracies[k]:
                print('k = %d, accuracy = %f' % (k, accuracy))

        # Plot the graph representing the mean and std deviation of accuracies for different k values
        accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
        accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
        plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
        plt.title('Cross-validation on k')
        plt.xlabel('k')
        plt.ylabel('Cross-validation accuracy')
        plt.show()

        return best_model

    def predict(self, model, x_test):
        return model.predict(x_test)

    def get_accuracy(self, y_test, y_pred):
        num_test = y_pred.shape[0]
        num_correct = np.sum(y_pred == y_test)
        accuracy = float(num_correct) / num_test
        return accuracy
