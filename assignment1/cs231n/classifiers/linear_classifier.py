import numpy as np
import itertools
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier(object):
    def __init__(self):
        self.W = None # A (D, C) matrix of weights
        self.learning_rate = 0
        self.regularization = 0


    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
    batch_size=200, verbose=False):
        self.learning_rate = learning_rate
        self.regularization = reg
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in xrange(num_iters):
            batch_idxs = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_idxs]
            y_batch = y[batch_idxs]
            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # Update the weights using the gradient and the learning rate.
            self.W += -learning_rate * grad

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[1])
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        pass

    @classmethod
    def getBestClassifier(cls, X, y, X_val, y_val, learning_rates, regularization_strengths, iterations):
        results = {}
        best_val = -1   # The highest validation accuracy that we have seen so far.
        best_classifier = None # The best classifier we have seen so far
        for lr,r in itertools.product(learning_rates, regularization_strengths):
            cur = cls()
            cur_best_accu = -1
            loss_hist = cur.train(X, y, learning_rate=lr, reg=r,
            num_iters=iterations, verbose=False)
            y_train_pred = cur.predict(X)
            y_train_accu = np.mean(y_train_pred == y)
            y_val_pred = cur.predict(X_val)
            y_val_accu = np.mean(y_val_pred == y_val)

            results[(lr, r)] = (y_train_accu, y_val_accu)
            if best_val < y_val_accu:
                best_val = y_val_accu
                best_classifier = cur
            print("learning rate: {}  reg: {}  best_accu: {}".format(lr, r, y_val_accu))

        return results, best_val, best_classifier

class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """
    def predict(self, X):
        # X - (N, D) matrix
        # self.W - (D, C) matrix of weights
        y = X.dot(self.W) # (N, C) matrix of results
        y_pred = np.argmax(y, 1) # N array of indices of the highest score
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def predict(self, X):
        print("softmax_loss_vectorized predict")

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
