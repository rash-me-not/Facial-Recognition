import numpy as np

class LinearClassifier(object):
    def __init__(self, data):
        self.W = None
        self.data = data
        self.x_train = self.data["x_train"]
        self.y_train = self.data["y_train"]
        self.x_val = self.data["x_val"]
        self.y_val = self.data["y_val"]
        self.x_test = self.data["x_test"]
        self.y_test = self.data["x_test"]

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=128, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            # Get random indices for batches
            batch_ind = np.random.choice(num_train, batch_size)
            X_batch = X[batch_ind]
            y_batch = y[batch_ind]

            # Evaluate loss and gradient
            #loss, grad = self.svm_loss_vectorized(self.W, X_batch, y_batch, reg)
            loss, grad = self.loss(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W += - learning_rate * grad

            # Perform parameter update
            if verbose and it % 10 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def train_and_validate(self):
        loss_history = self.train(self.x_train, self.y_train, verbose=True)
        y_pred = self.predict(self.x_train)
        num_corr_pred = np.sum(y_pred == self.y_train)
        acc_model = num_corr_pred / len(self.y_train)

        print('Linear model accuracy = %f' % (acc_model))

        return acc_model

    def loss(self, W, X_batch, y_batch, reg):
        """Simple loss function and gradient descent"""
        num_classes = W.shape[1]
        num_train = X_batch.shape[0]

        scores = X_batch.dot(W)
        margins = np.maximum(0, scores-scores[y_batch]+1)
        margins[y_batch] = 0
        loss = np.sum(margins)

        # grad
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
        dW = np.zeros(W.shape)  # initialize the gradient as zero
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

        X_mask = np.zeros(s.shape)
        X_mask[s > 0] = 1
        X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
        dW = X.T.dot(X_mask)
        dW /= num_train
        dW += 2 * reg * W

        return loss, dW

    def predict(self, X):
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred