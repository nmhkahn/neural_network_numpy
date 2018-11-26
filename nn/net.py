import numpy as np
from nn.layers import *

class TwoLayerNet:
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
        # TODO: #
        # #
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
        scores = None
        ######################################################################
        # TODO: #
        # #
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
        # TODO: #
        # #
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
        # TODO: #
        # #
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
        scores = None
        ######################################################################
        # TODO: #
        # #
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
        # TODO: #
        # #
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
    def __init__(
        self, 
        input_dim=(3,32,32), 
        num_filters=32,
        ksize=3, stride=1, pad=1,
        hidden_dim=100, 
        num_classes=10,
        init_mode="linear",
        init_scale=1e-3
):
        self.modules = dict()
        ######################################################################
        # TODO: #
        # #
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
        scores = None
        ######################################################################
        # TODO: #
        # #
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
        # TODO: #
        # #
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
