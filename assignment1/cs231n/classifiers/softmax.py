from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0
    # Iterate over all traning samples
    for i in range(X.shape[0]):
        # Get the scores for the current training sample
        scores = X[i].dot(W) # c x 1

        # Get the score of the correct prediction
        # and sum of scores (raised to the exp.)
        exp_correct = 0.0
        exp_sum = 0.0
        
        for j in range(scores.shape[0]):
            exp_sum += np.exp(scores[j])

            if j == y[i]:
                exp_correct = np.exp(scores[j])
        
        loss += -np.log(exp_correct / exp_sum)

        # Compute grad. for incorrect predictions
        for j in range(scores.shape[0]):
            if j != y[i]:
                dW[:, j] += + 1.0 * X[i] * (np.exp(scores[j]) / exp_sum)

        # Compute grad. for correct predictions
        dW[:, y[i]] += -1.0 * X[i] * ((exp_sum -  exp_correct) / exp_sum)
        
        # dloss = 1.0
        # dexp_correct = -(exp_sum / exp_correct) * 1.0 * dloss
        # dexp_sum = - (exp_sum / exp_correct) * (-1.0 * exp_correct * exp_sum ** (-2)) * dloss
        # dscores = np.exp(scores) * dexp_sum + (np.exp(scores[j])) * dexp_correct
        # dW += np.expand_dims(X[i], axis=1) @ np.expand_dims(dscores, axis=1).T

    loss /= X.shape[0]
    loss += reg * np.sum(W * W)

    dW = dW / X.shape[0]
    dW = dW + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    counts = np.exp(scores)
    counts_sum = np.sum(counts, axis=1, keepdims=True)
    probs = counts / counts_sum
    correct_class_probs = probs[range(X.shape[0]), y]
    log_loss = -np.log(correct_class_probs)
    loss = log_loss.sum(0) / X.shape[0]
    loss += reg * np.sum(W * W)

    dloss = 1.0
    dW = (2 * reg * W) * dloss
    dlog_loss = (np.ones_like(log_loss) * 1.0 / X.shape[0])* dloss
    dcorrect_class_probs = (-1.0 / correct_class_probs) * dlog_loss
    temp = np.zeros_like(probs)
    temp[range(X.shape[0]), y] = 1.0
    dprobs = temp * np.expand_dims(dcorrect_class_probs, 1)
    dcounts = (1 / counts_sum * np.ones_like(counts)) * dprobs
    dcounts_sum = ((counts * (-1.0 * counts_sum ** (-2))) * dprobs).sum(1, keepdims=True)
    dcounts += np.ones_like(counts) * dcounts_sum
    dscores = np.exp(scores) * dcounts
    dW += X.T @ dscores
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
