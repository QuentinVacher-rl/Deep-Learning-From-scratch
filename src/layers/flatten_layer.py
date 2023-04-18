import numpy as np
from layers.layer import Layer


class Flatten(Layer):

    def __init__(self) -> None:
        """Init Dense Layer"""
        super().__init__(None, None, None)

        self.trainable = False


    

    def init_weight(self):
        self.nb_weight = 0

    def forward(self, values):
        if len(values.shape) == len(self.input_shape) + 1:
            return values.reshape(tuple([values.shape[0]]) + tuple(self.output_shape))
        return values.reshape(self.output_shape)


    def backward(self, output_values, input_values, loss):
        if len(output_values.shape) == len(self.output_shape) + 1:
            return loss.reshape(tuple([output_values.shape[0]]) + tuple(self.input_shape)), None
        return loss.reshape(self.input_shape), None

    def set_shape(self, new_input_shape):
        self.set_input_shape(new_input_shape)
        self.set_output_shape(np.array([np.product(new_input_shape)]))


        
