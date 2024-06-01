import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """
    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        #######################################################################
        # TODO: Initialize the weights and biases of a two-layer network.     #
        #######################################################################

        # Initialize weights and biases
        self.w1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.b2 = np.zeros(num_classes)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def parameters(self):
        params = None
        #######################################################################
        # TODO: Build a dict of all learnable parameters of this model.       #
        #######################################################################

        params = {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2
        }

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return params

    def forward(self, X):
        scores, cache = None, None
        #######################################################################
        # TODO: Implement the forward pass to compute classification scores   #
        # for the input data X. Store into cache any data that will be needed #
        # during the backward pass.                                           #
        #######################################################################

        # Forward pass
        out1, cache1 = fc_forward(X, self.w1, self.b1)
        out2, cache2 = relu_forward(out1)
        scores, cache3 = fc_forward(out2, self.w2, self.b2)
        cache = (cache1, cache2, cache3)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return scores, cache

    def backward(self, grad_scores, cache):
        grads = None
        #######################################################################
        # TODO: Implement the backward pass to compute gradients for all      #
        # learnable parameters of the model, storing them in the grads dict   #
        # above. The grads dict should give gradients for all parameters in   #
        # the dict returned by model.parameters().                            #
        #######################################################################

        # Unpack cache
        cache1, cache2, cache3 = cache

        # Backward pass
        grad_out2, grad_w2, grad_b2 = fc_backward(grad_scores, cache3)
        grad_out1 = relu_backward(grad_out2, cache2)
        grad_x, grad_w1, grad_b1 = fc_backward(grad_out1, cache1)

        # Store gradients
        grads = {
            'w1': grad_w1,
            'b1': grad_b1,
            'w2': grad_w2,
            'b2': grad_b2
        }

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return grads
