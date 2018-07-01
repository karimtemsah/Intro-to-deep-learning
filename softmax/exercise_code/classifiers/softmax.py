"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    epsilon = 1e-12
    y_pred = np.matmul(X, W)
    y_pred -= np.amax(y_pred,axis = 1,keepdims = True)
    reg_result = np.sum(np.square(W))
    for i in range(X.shape[0]):
        exp = 0
        for z in range(W.shape[1]):
            exp += np.exp(y_pred[i,z])
        for j in range(W.shape[1]):
            loss += (j==y[i]) * np.log((np.exp(y_pred[i,j]) / exp))
            p = np.exp(y_pred[i,j])/exp
            dW[:, j] += (p-(j == y[i])) * X[i, :]

            
    loss = -1 * loss * (1/y.shape[0]) + (reg/2) * reg_result
    dW = dW /y.shape[0]  + reg*W

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    epsilon = 1e-12
    y_pred = np.matmul(X, W)
    y_pred -= np.amax(y_pred,axis = 1,keepdims = True)
    reg_result = np.sum(np.square(W))
    target = np.zeros((y.shape[0], W.shape[1]))
    target[np.arange(y.shape[0]), y] = 1
    exp = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1)[:, None]
    loss = (-np.sum(target*np.log(exp))/y.shape[0]) + (reg/2) * reg_result

    
    exp[range(X.shape[0]),y] -= 1
    exp /= X.shape[0]
    dW = np.dot(X.T,exp)
    dW += reg * W
    
    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = 0
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 2e-7, 3e-7]
    regularization_strengths = [1e4, 2e4, 3e4, 4e4]
    best_lr = 0
    best_reg = 0
    besr_r = 0
    for lr in learning_rates:
        for r in regularization_strengths:
            softmax = SoftmaxClassifier()
            loss_hist = softmax.train(X_train, y_train, learning_rate=lr, reg=r, num_iters=1500, verbose=False)
            y_train_pred = softmax.predict(X_train)
            train_accuracy = np.mean(y_train == y_train_pred)
            y_val_pred = softmax.predict(X_val)
            val_accuracy = np.mean(y_val == y_val_pred)
            all_classifiers.append(softmax)
            print('lr: {}, r:{}, train_accuracy:{}, val_accuracy:{}'.format(lr, r, train_accuracy, val_accuracy))
            results[(lr, r)] = (train_accuracy, val_accuracy)
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_lr = lr
                best_r = r
                best_softmax = softmax
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
    
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
