from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

class KNNSklearn:

    def __init__(self, data):
        self.data = data
        self.x_train = self.data["x_train"]
        self.y_train = self.data["y_train"]
        self.x_val = self.data["x_val"]
        self.y_val = self.data["y_val"]
        self.x_test = self.data["x_test"]
        self.y_test = self.data["x_test"]

    def get_knn_model(self, k, dist_metric):
        """When p = 1, this is equivalent to using manhattan_distance , and euclidean_distance for p = 2"""
        dist_dict = {"Manhattan":1, "Euclidean":2}
        model = KNeighborsClassifier(n_neighbors=k, p=dist_dict[dist_metric])
        return model

    def train_and_validate(self, k_list, dist_metric):

        validation_accuracies = []
        for k in k_list:
            model = self.get_knn_model(k, dist_metric)
            model.fit(self.x_train, self.y_train)
            filename = 'finalized_model_' + str(k) + '_neighbor.sav'
            pickle.dump(model, open(filename, 'wb'))
            # model = pickle.load(open(filename, 'rb'))
            acc = model.score(self.x_val, self.y_val)
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



    def train_wd_cross_validation(self, num_folds, k_choices, dist_metric):

        X_train_folds = []
        y_train_folds = []
        # Step 5.2: split up the training data and labels into folds.
        X_train_folds = np.split(self.x_train, num_folds)
        y_train_folds = np.split(self.y_train, num_folds)

        # Step 5.3: perform k-fold cross validation to find the best value of k. For each k, run the KNN method multiple times. Store the accuracies for all folds and all k values.

        k_to_accuracies = {}  # k_to_accuracies[k]: a list giving the different accuracy values
        num_split = self.x_train.shape[0] / num_folds
        acc_k = np.zeros((len(k_choices), num_folds), dtype=np.float)
        # call classifiers multiple times to generate the cross-validation
        for ik, k in enumerate(k_choices):
            for i in range(num_folds):
                train_set = np.concatenate((X_train_folds[:i] + X_train_folds[i + 1:]))
                label_set = np.concatenate((y_train_folds[:i] + y_train_folds[i + 1:]))
                model = self.get_knn_model(k, dist_metric)
                model.train(train_set, label_set)
                y_pred_fold = model.predict(X_train_folds[i], k=k, num_loops=0)
                num_correct = np.sum(y_pred_fold == y_train_folds[i])
                acc_k[ik, i] = float(num_correct) / num_split
            k_to_accuracies[k] = acc_k[ik]

        # Print out the computed accuracies
        for k in sorted(k_to_accuracies):
            for accuracy in k_to_accuracies[k]:
                print('k = %d, accuracy = %f' % (k, accuracy))

    def predict(self):
        pass

