import numpy as np
from nn.layers import *

class TwoLayerNet:
    """A neural network that has two fully-connected (linear) layers.
    This model can be illustrated as:
    `input -> linear@hidden_dim -> relu -> linear@num_classes -> softmax`
    Here, linear@X represents linear layer that has `X` output dimension.

    Args:
        - input_dim (int): Input dimension.
        - hidden_dim (int): Hidden dimension. 
          It should be output dimension of first linear layer.
        - num_classes (int): Number of classes, and it should be output 
          dimension of second (last) linear layer.
        - init_mode (str): Weight initalize method. See `nn.init.py`.
          linear|normal|xavier|he are the possible options.
        - init_scale (float): Weight initalize scale for the normal init way.
          See `nn.init.py`.
    """
    def __init__(
        self, 
        input_dim=3*32*32, 
        hidden_dim=100, 
        num_classes=10,
        init_mode="linear",
        init_scale=1e-3
):
        self.modules = dict()
        ######################################################################
        # TODO: Initalize all the needed modules in this network which is:   #
        # input -> linear -> relu -> linear -> softmax.                      #
        ######################################################################
        self.modules["linear1"] = Linear(
            input_dim, hidden_dim, 
            init_mode, init_scale
        )
        self.modules["linear2"] = Linear(
            hidden_dim, num_classes, 
            init_mode, init_scale
        )
        self.modules["relu1"] = ReLU()
        self.modules["softmax"] = SoftmaxCELoss()
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    
    def loss(self, X, y=None):
        """Compute loss and gradient for a minibatch of data.

        Args:
            - X: Array of input data of shape (N, C), where N is batch size 
              and C is input_dim.
            - y: Array of labels of shape (N,). y[i] gives the label for X[i].

        Return:
            - loss: Loss for a current minibatch of data.
        """
        scores = None
        ######################################################################
        # TODO: Implement forward propagation. First, you should calculate   #
        # scores which is pre-activation value of softmax, then store into   #
        # scores variable.                                                   #
        ######################################################################
        out = self.modules["linear1"].forward(X)
        out = self.modules["relu1"].forward(out)
        scores = self.modules["linear2"].forward(out)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        if y is None:
            return scores
        ######################################################################
        # TODO: Implement backward propagation using the loss from the       #
        # softmax cross-entropy layer.                                       #
        ######################################################################
        loss, dout = self.modules["softmax"].forward(scores, y)
        dout = self.modules["linear2"].backward(dout)
        dout = self.modules["relu1"].backward(dout)
        dout = self.modules["linear1"].backward(dout)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss


class FCNet:
    """A neural network that has arbitrary number of layers.
    This model can be illustrated as:
    `input -> linear -> relu -> linear -> ... -> linear -> softmax`

    Args:
        - hidden_dims (list): Hidden dimensions of layers.
          Each element are the output dimension of i-th fc layer.
          So that, total #layers = len(hidden_dims) + 1
        - Other arguments are same as the TwoLayerNet.
    """
    def __init__(
        self, 
        input_dim=3*32*32, 
        hidden_dims=[100, 100, 100], 
        num_classes=10,
        init_mode="linear",
        init_scale=1e-3
):
        self.modules = dict()
        self.num_layers = 1 + len(hidden_dims)
        ######################################################################
        # TODO: Initalize the FCNet that has arbitrary number of layer.      #
        # All the layers have to be stored in the self.modules with proper   #
        # name as a key (e.g. "fc1").                                        #
        ######################################################################
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.modules["linear"+str(i+1)] = Linear(
                dims[i], dims[i+1],
                init_mode, init_scale
            )
            
            # no activation after the last layer
            if i < self.num_layers-1:
                self.modules["relu"+str(i+1)] = ReLU()
        self.modules["softmax"] = SoftmaxCELoss()
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    
    def loss(self, X, y=None):
        """Compute loss and gradient for a minibatch of data.
        Args and Returns are same as the TwoLayerNet.
        """
        scores = None
        ######################################################################
        # TODO: Implement forward propagation. First, you should calculate   #
        # scores which is pre-activation value of softmax, then store into   #
        # scores variable.                                                   #
        ######################################################################
        out = X
        for i in range(self.num_layers-1):
            out = self.modules["linear"+str(i+1)].forward(out)
            out = self.modules["relu"+str(i+1)].forward(out)
        scores = self.modules["linear"+str(self.num_layers)].forward(out)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        if y is None:
            return scores
        ######################################################################
        # TODO: Implement backward propagation using the loss from the       #
        # softmax cross-entropy layer.                                       #
        ######################################################################
        loss, dout = self.modules["softmax"].forward(scores, y)
        dout = self.modules["linear"+str(self.num_layers)].backward(dout)
        for i in reversed(range(self.num_layers-1)):
            dout = self.modules["relu"+str(i+1)].backward(dout)
            dout = self.modules["linear"+str(i+1)].backward(dout)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss
        
        
class ThreeLayerConvNet:
    """A neural network that has one conv and two linear layers.
    This model can be illustrated as:
    `input -> conv@num_filters -> relu -> pool -> linear@hidden_dim ->
    relu -> linear@num_classes -> softmax`.
    Here, linear@X represents linear layer that has `X` output dimension and
    conv@X shows conv layer with `X` number of filters.

    Unlike FCNet, the network operates on minibatches of data have 
    shape (N, C, H, W) consisting of N images (batch size), each with 
    height H and width W and with C input channels.

    Args:
        - input_dim (list or tuple): Input dimension of single input 
          **image**. Normally, it could be (C, H, W) dimension.
        - num_filters (int): Number of filters (channels) of conv layer.
        - ksize (int): Kernel size of conv layer.
        - stride (int): Stride of conv layer.
        - pad (int): Number of padding of conv layer.
        - Other arguments are same as the TwoLayerNet.
    """
    def __init__(
        self, 
        input_dim=[3,32,32], 
        num_filters=32,
        ksize=3, stride=1, pad=1,
        hidden_dim=100, 
        num_classes=10,
        init_mode="linear",
        init_scale=1e-3
):
        self.modules = dict()
        ######################################################################
        # TODO: Initalize the three-layer CNN which has: input -> conv ->    #
        # relu -> pool -> linear -> relu -> linear -> softmax.               #
        # All the layers have to be stored in the self.modules with proper   #
        # name as a key (e.g. "conv1").                                      #
        ######################################################################
        self.modules["conv1"] = Conv2d(
            input_dim[0], num_filters,
            ksize=ksize, stride=stride, pad=pad,
            init_mode=init_mode, init_scale=init_scale
        )
        self.modules["relu1"] = ReLU()
        self.modules["pool1"] = MaxPool2d(2, 2)

        h, w = input_dim[1]//2, input_dim[2]//2

        self.modules["linear1"] = Linear(
            num_filters*h*w, hidden_dim,
            init_mode, init_scale
        )
        self.modules["relu2"] = ReLU()
        self.modules["linear2"] = Linear(
            hidden_dim, num_classes,
            init_mode, init_scale
        )
        self.modules["softmax"] = SoftmaxCELoss()
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    
    def loss(self, X, y=None):
        """Compute loss and gradient for a minibatch of data.

        Args:
            - X: Array of input **image** data of shape (N, C, H, W), where
              N is batch size, C is number of channels, height and for width.
            - y: Array of labels of shape (N,). y[i] gives the label for X[i].

        Return:
            - loss: Loss for a current minibatch of data.
        """
        scores = None
        ######################################################################
        # TODO: Implement forward propagation. First, you should calculate   #
        # scores which is pre-activation value of softmax, then store into   #
        # scores variable.                                                   #
        ######################################################################
        out = X
        out = self.modules["conv1"].forward(out)
        out = self.modules["relu1"].forward(out)
        out = self.modules["pool1"].forward(out)

        shape = out.shape
        out = out.reshape(shape[0], -1)
        
        out = self.modules["linear1"].forward(out)
        out = self.modules["relu2"].forward(out)
        scores = self.modules["linear2"].forward(out)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        if y is None:
            return scores
        ######################################################################
        # TODO: Implement backward propagation using the loss from the       #
        # softmax cross-entropy layer.                                       #
        ######################################################################
        loss, dout = self.modules["softmax"].forward(scores, y)
        dout = self.modules["linear2"].backward(dout)
        dout = self.modules["relu2"].backward(dout)
        dout = self.modules["linear1"].backward(dout)
        dout = dout.reshape(shape)
        
        dout = self.modules["pool1"].backward(dout)
        dout = self.modules["relu1"].backward(dout)
        dout = self.modules["conv1"].backward(dout)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss
