from object import Value
import numpy as np

class Neural_Network:
    def __init__(self , input_size , layer_sizes):
        self.layers = []
        sizes = [input_size] + list(layer_sizes)  #sizes = [3, 4, 1]

        for nin , nout in zip(sizes , sizes[1:]):
            layer = []
            for _ in range(nout):
                weights = [Value(np.random.uniform(-1 , 1)) for _ in range(nin)]
                bias = Value(np.random.uniform(-1,1))
                layer.append((weights , bias))

            self.layers.append(layer)

    def __call__(self , x):
        out = x
        for layer in self.layers:
            next_out = []

            for weights , bias in layer:
                activation = bias
                for wi , xi in zip(weights , out):
                    activation = activation  + wi * xi

                next_out.append(activation.tanh())
            out = next_out[0] if len(next_out) == 1 else next_out
        return out

    def parameters(self):
        params = []
        for layer in self.layers:
            for weights , bias in layer:
                params.extend(weights)
                params.append(bias)
        return params
        