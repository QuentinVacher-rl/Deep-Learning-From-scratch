
from typing import Iterable
import math

import numpy as np


class Layer:

    def __init__(self, output_shape: Iterable, activation_function="ReLu|Softmax", input_shape:Iterable=None, batched=False) -> None:
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation_function=activation_function

        self.desactivated_during_test = False
        self.trainable = True

        # Classic initialiser
        self.weight = np.ndarray
        self.nb_weight = 0

    def init_weight(self):
        pass

    def init_weight_method(self, array):
        return (array *2-1) * math.sqrt(2/(np.product(self.input_shape)))

    def forward(self, value):
        pass

    def backward(self, output_values, input_values, loss):
        pass

    def activate_function(self, values):
        if self.activation_function == "ReLu":
            values[values<0] = 0
            return values

        if self.activation_function == "leaky ReLu":
            values = np.maximum(values, 0.01 * values)
            return values

        if self.activation_function == "Sigmoid":
            values = 1 / (1 + np.exp(-1 * values))
            return values

    def derivate_function(self, values):
        if self.activation_function == "ReLu":
            values[values<0] = 0
            values[values>0] = 1
            return values

        if self.activation_function == "leaky ReLu":
            values[values<0] = 0.01
            values[values>0] = 1
            return values


        if self.activation_function == "Sigmoid":
            values = values * (1-values)
            return values

    def set_shape(self, new_input_shape):
        self.set_input_shape(new_input_shape)
        pass

    def set_output_shape(self, new_output_shape):
        self.output_shape = new_output_shape

    def set_input_shape(self, new_input_shape):
        self.input_shape = new_input_shape

    def has_input_shape(self):
        return not self.input_shape is None
    
    def control_input_and_output_shape(self):
        pass

    