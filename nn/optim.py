import numpy as np
import nn.layers as layers

class Optimizer:
    def __init__(self, net):
        if not isinstance(net, (list, dict)):
            raise ValueError(
                """net argument has to be list
                 or dict of layer.Layer class instance."""
            )
        if isinstance(net, list):
            for module in net:
                if not isinstance(module, layers.Layer):
                    raise ValueError(
                        """Elements of net argument have to be
                        layers.Layer or inherited class instance."""
                    )
        else:
            for module in net.values():
                if not isinstance(module, layers.Layer):
                    raise ValueError(
                        """Elements of net argument have to be
                        layers.Layer or inherited class instance."""
                    )

        self.net = net
        
    def step(self):
        if isinstance(self.net, list):
            for module in self.net:
                if hasattr(module, "params"):
                    self._update(module.params, module.grads)
        else:
            for k, v in self.net.items():
                if hasattr(self.net[k], "params"):
                    self._update(self.net[k].params, self.net[k].grads)
     
    def _update_module(self, param, grad):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, net, lr=0.001):
        super().__init__(net)

        self.lr = lr

    def _update(self, params, grads):
        for k in params.keys():
            params[k] -= grads[k] * self.lr
