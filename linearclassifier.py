import numpy as np

class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-7, reg=1e-5, num_iters=1000,
              batch_size=128, verbose=False, loss_function="svm"):
        """
        Train a linear classifier with gradient descent. Training set is divided in batches.
        Matrix W of weights is initialized with random values, and model keeps W with
        minimal loss.

        Inputs:
        - X: numpy.ndarray of shape (N, D) containing training data; N is number the 
             of training samples, D is number of pixels in each image (height * width), 
        - y: numpy.ndarray of shape (N,) containing training labels (correct class of each image)
        - learning rate: (float) learning rate for optimization
        - reg: (float) regularization strength
        - batch_size: (integer) number of training samples to use per batch
        - verbose: (boolean) If true, print progress during training
        - loss_function: which method to use to calculate the loss and gradient descent of the model.
            options are 'gd' for simple gradient descent, 'svm_loss_naive' for naive implementation
            of SVM, and 'svm' for vectorized SVM implementation.

        Output:
        - loss_history: array containing the loss obtained at each iteration
        """

        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            # lazily initialize matrix of weights W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        W_history = []
        for it in range(num_iters):
            # Get random indices for batches
            batch_ind = np.random.choice(num_train, batch_size)
            X_batch = X[batch_ind]
            y_batch = y[batch_ind]

            # Evaluate loss and gradient
            if loss_function == "gd":
                loss, grad = self.loss(self.W, X_batch, y_batch, reg)
            elif loss_function == "svm":
                loss, grad = self.svm_loss_vectorized(self.W, X_batch, y_batch, reg)
            elif loss_function == "naive_svm":
                loss, grad = self.svm_loss_naive(self.W, X_batch, y_batch, reg)
            else:
                raise ValueError('Invalid value %d for loss function' % loss_function)
            
            loss_history.append(loss)
            W_history.append(self.W)

            alpha = learning_rate * grad
            self.W = self.W - alpha

            # Perform parameter update
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        # Keep W obtained with minimal loss
        self.W = W_history[np.argmin(loss_history)]
        return loss_history

    def train_and_validate(self, x_train, y_train, x_val, y_val, loss_function="svm"):
        """
        Train a linear classifier on training and validation sets. 
        """
        loss_history = self.train(x_train, y_train, loss_function=loss_function, verbose=True)

        y_train_pred = self.predict(x_train)
        num_corr_training_pred = np.sum(y_train_pred == y_train)
        train_acc_model = num_corr_training_pred / len(y_train)

        y_val_pred = self.predict(x_val)
        num_corr_val_pred = np.sum(y_val_pred == y_val)
        val_acc_model = num_corr_val_pred / len(y_val)

        return train_acc_model, val_acc_model

    def get_accuracy(self, y_pred, y):
        """Calculate accuracy of predictions over ground truth."""
        num_corr_pred = np.sum(y_pred == y)
        acc_model = num_corr_pred / len(y)

        print('Linear model accuracy = %f' % (acc_model))

        return acc_model


    def loss(self, W, X_batch, y_batch, reg):
        """Simple loss function with gradient descent"""
        num_classes = W.shape[1]
        num_train = X_batch.shape[0]

        # get prediction values with current W to calculate loss
        scores = X_batch.dot(W)
        margins = []
        for i in range(num_train):
            margins.append(np.maximum(0, scores[i] - scores[i][y_batch[i]]+1))

        loss = np.sum(margins) - num_train

        # calculate gradient descent
        X_mask = np.zeros(scores.shape)
        X_mask[scores > 0] = 1
        X_mask[np.arange(num_train), y_batch] = -np.sum(X_mask, axis=1)
        dW = X_batch.T.dot(X_mask)
        dW /= num_train
        dW += 2 * reg * W

        return loss, dW

    def svm_loss_vectorized(self, W, X, y, reg):
        """Structured SVM loss function, vectorized implementation"""
        loss = 0.0
        num_train = X.shape[0]

        # s: A numpy array of shape (N, C) containing scores
        s = X.dot(W)

        # read correct scores into a column array of height N
        correct_score = s[list(range(num_train)), y]
        correct_score = correct_score.reshape(num_train, -1)

        # subtract correct scores from score matrix and add margin
        s += 1 - correct_score
        
        # make sure correct scores themselves don't contribute to loss function
        s[list(range(num_train)), y] = 0
        
        # construct loss function
        loss = np.sum(np.fmax(s, 0)) / num_train
        loss += reg * np.sum(W * W)

        # calculate gradient descent
        X_mask = np.zeros(s.shape)
        X_mask[s > 0] = 1
        X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
        dW = X.T.dot(X_mask)
        dW /= num_train
        dW += 2 * reg * W

        return loss, dW

    def svm_loss_naive(self, W, X, y, reg):
        """Structured SVM loss function, naive implementation with loops."""
        dW = np.zeros(W.shape)

        # compute the loss and the gradient
        num_classes = W.shape[1]
        num_train = X.shape[0]
        loss = 0.0

        for i in range(num_train):
            scores = X[i].dot(W)
            correct_class_score = scores[y[i]]
            loss_contributors_count = 0
            for j in range(num_classes):
                if j == y[i]:
                    continue
                margin = scores[j] - correct_class_score + 1  # note delta = 1
                if margin > 0:
                    loss += margin
                    dW[:, j] += X[i]
                    # count contributor terms to loss function
                    loss_contributors_count += 1
            dW[:, y[i]] += (-1) * loss_contributors_count * X[i]

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_train
        dW /= num_train

        # Add regularization to the loss.
        loss += reg * np.sum(W * W)
        # Add regularization to the gradient
        dW += 2 * reg * W

        return loss, dW


    def predict(self, X):
        """Use the weights matrix W to predict labels for data points."""
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred