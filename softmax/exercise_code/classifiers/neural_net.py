"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        layer_1 = np.matmul(X,W1) + b1
        layer_2 = np.maximum(layer_1, 0)
        scores = np.matmul(layer_2, W2) + b2

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        scores -= np.amax(scores,axis = 1,keepdims = True)
        reg_result_1 = np.sum(np.square(W2))
        reg_result_2 = np.sum(np.square(W1))
        target = np.zeros((y.shape[0], W2.shape[1]))
        target[np.arange(y.shape[0]), y] = 1
        exp = np.exp(scores) / (np.sum(np.exp(scores), axis=1)[:, None])
        loss = (-np.sum(target*np.log(exp))/y.shape[0]) + (reg/2) * reg_result_1 + (reg/2) * reg_result_2
        

        # Backward pass: compute gradients
        grads = {}
        exp[range(X.shape[0]),y] -= 1
        exp /= X.shape[0]
        grads['W2'] = np.matmul(layer_2.T, exp) + reg * W2
        grads['b2'] = np.sum(exp, axis=0)
        grad_layer2 = np.matmul(exp, W2.T)
        grad_layer1 = grad_layer2 * (layer_1 >=0)
        grads['W1'] = np.matmul(X.T, grad_layer1) + reg * W1
        grads['b1'] = np.sum(grad_layer1, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            indicies = np.random.choice(X.shape[0], batch_size)
            X_batch = X[indicies]
            y_batch = y[indicies]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            self.params['W1'] -=  learning_rate * grads['W1']
            self.params['b1'] -=  learning_rate * grads['b1']
            self.params['W2'] -=  learning_rate * grads['W2']
            self.params['b2'] -=  learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################
        y_pred = np.argmax(self.loss(X), axis=1)
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 
    best_val = 0
    learning_rates = [1e-3]
    regularization_strengths = [1e-2]
    best_lr = 0
    best_reg = 0
    besr_r = 0
    input_size = X_train.shape[1]
    num_classes = 10
    hiddine_layers = [70, 80]
    batch_size = [250, 300, 350, 400]
    for lr in learning_rates:
        for r in regularization_strengths:
            for layer in hiddine_layers:
                for batch in batch_size:
                    network = TwoLayerNet(input_size,layer,num_classes)
                    loss_hist = network.train(X_train, y_train, X_val, y_val, learning_rate=lr, reg=r, num_iters=1500, batch_size=batch, verbose=False)
                    y_train_pred = network.predict(X_train)
                    train_accuracy = np.mean(y_train == y_train_pred)
                    y_val_pred = network.predict(X_val)
                    val_accuracy = np.mean(y_val == y_val_pred)
                    print('lr: {}, r:{}, layers: {}, batch:{}, train_accuracy:{}, val_accuracy:{}'.format(lr, r, layer, batch, train_accuracy, val_accuracy))
                    if val_accuracy > best_val:
                        best_val = val_accuracy
                        best_lr = lr
                        best_r = r
                        best_net = network
    
    print('best validation accuracy achieved during validation: %f' % best_val)
    return best_net
