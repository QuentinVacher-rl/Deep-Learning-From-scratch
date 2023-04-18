from typing import Iterable
import numpy as np
from layers.layer import Layer


class Dropout(Layer):

    def __init__(self, coef=0.2) -> None:
        """Init Dense Layer"""
        super().__init__(None, None, None)

        self.dropout_coef = 0.2
        self.mask = np.ndarray
        self.desactivated_during_test = True
        self.trainable = False


    

    def init_weight(self):
        self.nb_weight = 0

    def forward(self, values):
        filter = np.random.random(values.shape) 
        self.mask = np.zeros(values.shape)
        self.mask[filter > self.dropout_coef] = 1 / (1-self.dropout_coef)
        return values * self.mask


    def backward(self, output_values, input_values, loss):
        return loss * self.mask, None

    def set_shape(self, new_input_shape):
        self.set_input_shape(new_input_shape)
        self.set_output_shape(new_input_shape)

        
