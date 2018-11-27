import numpy as np
from nn.init import initialize

class Layer:
    """Base class for all neural network modules.
    You must implement forward and backward method to inherit this class.
    All the trainable parameters have to be stored in params and grads to be
    handled by the optimizer.
    """
    def __init__(self):
        self.params, self.grads = dict(), dict()

    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *input):
        raise NotImplementedError


class Linear(Layer):
    """Linear (fully-connected) layer.

    Args:
        - in_dims (int): 
        
    """
    def __init__(self, in_dims, out_dims, init_mode="linear", init_scale=1e-3):
        super().__init__()

        self.params["w"] = initialize((in_dims, out_dims), init_mode, init_scale)
        self.params["b"] = initialize(out_dims, "zero")
    
    def forward(self, x):
        ######################################################################
        # TODO: Linear 레이어의 forward propagation 구현.
        ######################################################################
        self.x = x[:]
        out = x.dot(self.params["w"]) + self.params["b"].T
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Linear 레이어의 backward propagation 구현.
        ######################################################################
        x = self.x
        dx = dout.dot(self.params["w"].T).reshape(x.shape)
        dw = x.T.dot(dout)
        db = np.sum(dout.T, axis=1)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        self.grads["w"] = dw
        self.grads["b"] = db
        return dx


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: ReLU 레이어의 forward propagation 구현.
        ######################################################################
        self.x = x[:]
        out = np.clip(x, 0.0, None)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: ReLU 레이어의 backward propagation 구현.
        ######################################################################
        x = self.x
        out = np.clip(x, 0.0, None)
        out[out>0] = 1

        dx = out*dout
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: Sigmoid 레이어의 forward propagation 구현.
        ######################################################################
        self.x = x[:]
        out = 1/(1+np.exp(-x))
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Sigmoid 레이어의 backward propagation 구현.
        ######################################################################
        sigmoid = self.forward(self.x)
        dx = sigmoid*(1-sigmoid)*dout
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ######################################################################
        # TODO: Tanh 레이어의 forward propagation 구현.
        ######################################################################
        self.x = x[:]
        sigmoid2x = 1/(1+np.exp(-2*x))
        out = 2*sigmoid2x-1
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Tanh 레이어의 backward propagation 구현.
        ######################################################################
        tanh = self.forward(self.x)
        dx = (1-tanh**2)*dout
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx


class SoftmaxCELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        ######################################################################
        # TODO: Softmax cross-entropy 레이어의 구현. 
        #        
        # NOTE: 이 메소드에서 forward/backward를 모두 수행하고, loss와 gradient (dx)를 
        # 리턴해야 함.
        ######################################################################
        N = x.shape[0]
        
        probs = np.exp(x-np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss, dx
    
    
class Conv2d(Layer):
    def __init__(
        self, 
        in_dims, out_dims,
        ksize, stride, pad,
        init_mode="linear",
        init_scale=1e-3
    ):
        super().__init__()
        
        self.params["w"] = initialize(
            (out_dims, in_dims, ksize, ksize), 
            init_mode, init_scale)
        self.params["b"] = initialize(out_dims, "zero")
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        ######################################################################
        # TODO: Convolution 레이어의 forward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################
        self.x = x[:]
        
        N, C, H, W = x.shape
        F, HH, WW = self.out_dims, self.ksize, self.ksize
        stride, pad = self.stride, self.pad

        # padding
        pad_x = np.pad(
            x, 
            ((0, 0), (0, 0), (pad, pad), (pad, pad)), 
            "constant", constant_values=0
        )
        # calculate output shape
        H_ = 1 + (H + 2 * pad - HH) // stride
        W_ = 1 + (W + 2 * pad - WW) // stride
        out = np.zeros((N, F, H_, W_))

        # really dirty code :(
        for n in range(N):
            for f in range(F):
                for j in range(H_):
                    for i in range(W_):
                        source = pad_x[n, :, j*stride:j*stride+HH, i*stride:i*stride+WW]
                        filter = self.params["w"][f, :, :, :]
                        out[n, f, j, i] = np.sum(source*filter) + self.params["b"][f]
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Convolution 레이어의 backward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################
        x, w, b  = self.x, self.params["w"], self.params["b"]
        stride, pad = self.stride, self.pad
          
        # padding
        pad_x = np.pad(
            x, 
            ((0, 0), (0, 0), (pad, pad), (pad, pad)), 
            "constant", constant_values=0
        )
        N, C, H, W = x.shape
        F, HH, WW = w.shape[0], w.shape[2], w.shape[3]
        H_ = 1 + (H + 2 * pad - HH) // stride
        W_ = 1 + (W + 2 * pad - WW) // stride
  
        dx = np.zeros(pad_x.shape)
        dw = np.zeros(w.shape)
        db = np.zeros(b.shape)

        # what a dirty code! :(
        for n in range(N):
            for f in range(F):
                for j in range(H_):
                    for i in range(W_):
                        dx[n, :, j*stride:j*stride+HH, i*stride:i*stride+WW] += np.dot(w[f, :, :, :], dout[n, f, j, i])
                        dw[f, :, :, :] += np.dot(pad_x[n, :, j*stride:j*stride+HH, i*stride:i*stride+WW], dout[n, f, j, i])
                        db[f] += np.sum(dout[n, f, j, i])

        # remove padding
        dx = dx[:, :, 1:dx.shape[2]-1, 1:dx.shape[3]-1]
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        self.grads["w"] = dw
        self.grads["b"] = db
        return dx
    

class MaxPool2d(Layer):
    def __init__(self, ksize, stride):
        super().__init__()
        
        self.ksize = ksize
        self.stride = stride
        
    def forward(self, x):
        ######################################################################
        # TODO: Max pooling 레이어의 forward propagation 구현.
        #
        # HINT: for-loop의 2-중첩으로 구현.
        ######################################################################
        self.x = x[:]
        pool_H, pool_W, stride = self.ksize, self.ksize, self.stride
        N, C, H, W = x.shape
        H_ = (H-pool_H) // stride + 1
        W_ = (W-pool_W) // stride + 1

        out = np.zeros((N, C, H_, W_))
        for j in range(H_):
            for i in range(W_):
                out[:,:,j,i] = np.max(x[:,:,j*stride:j*stride+pool_H,i*stride:i*stride+pool_W], axis=(2,3))
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out

    def backward(self, dout):
        ######################################################################
        # TODO: Max pooling 레이어의 backward propagation 구현.
        #
        # HINT: for-loop의 4-중첩으로 구현.
        ######################################################################
        x, pool_H, pool_W, stride = self.x, self.ksize, self.ksize, self.stride
        N, C, H, W = x.shape
        H_ = (H-pool_H) // stride + 1
        W_ = (W-pool_W) // stride + 1

        dx = np.zeros(x.shape)
        for n in range(N):
            for c in range(C):
                for j in range(H_):
                    for i in range(W_):
                        argmax = np.argmax(x[n, c, j*stride:j*stride+pool_H, i*stride:i*stride+pool_W])
                        h, w = divmod(argmax, pool_H)
                        dx[n, c, j*stride+h, i*stride+w] = dout[n, c, j, i]
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.grads["x"] = dx
        return dx
