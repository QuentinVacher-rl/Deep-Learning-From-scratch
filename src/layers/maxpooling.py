import math
from tkinter import E
import numpy as np
from layers.layer import Layer

from scipy import signal
from typing import Iterable

class MaxPooling(Layer):

    def __init__(self) -> None:
        """Init Dense Layer"""
        super().__init__(None, None, None)
        self.trainable = False
        self.del_axis_1 = False
        self.del_axis_2 = False


    def init_weight(self):
        self.nb_weight = 0


    def forward(self, values):
        if len(values.shape) == 4:        
            if self.del_axis_1:
                values = np.delete(values, (-1), axis=2)
            if self.del_axis_2:
                values = np.delete(values, (-1), axis=3)

            shape = values.shape
            first_max_shape = (shape[0], shape[1], shape[2], shape[3]//2, 2)
            second_max_shape = (shape[0], shape[1], shape[2]//2, shape[3]//2, 2)

            values = np.transpose(np.max(
                np.transpose(np.max(
                    values.reshape(first_max_shape), axis=4
                    ) ,axes=(0,1,3,2)).reshape(second_max_shape), axis=4
            ), axes=(0,1,3,2))

        else:
            if self.del_axis_1:
                values = np.delete(values, (-1), axis=1)
            if self.del_axis_2:
                values = np.delete(values, (-1), axis=2)

            shape = values.shape
            first_max_shape = (shape[0], shape[1], shape[2]//2, 2)
            second_max_shape = (shape[0], shape[1]//2, shape[2]//2, 2)

            values = np.transpose(np.max(
                np.transpose(np.max(
                    values.reshape(first_max_shape), axis=3
                    ) ,axes=(0,2,1)).reshape(second_max_shape), axis=3
            ), axes=(0,2,1))

        return values


    def backward(self, output_values, input_values, loss):

        if len(output_values.shape) == 4:
            output_values = np.repeat(np.repeat(output_values, 2, axis=2), 2, axis=3)
            loss = np.repeat(np.repeat(loss, 2, axis=2), 2, axis=3)

            if self.del_axis_1:
                added_values = np.zeros((output_values.shape[0], output_values.shape[1], 1, output_values.shape[3]))
                output_values = np.concatenate((output_values, added_values), axis=2)
                loss = np.concatenate((loss, added_values), axis=2)
            if self.del_axis_2:
                added_values = np.zeros((output_values.shape[0], output_values.shape[1], output_values.shape[2], 1))
                output_values = np.concatenate((output_values, added_values), axis=3)
                loss = np.concatenate((loss, added_values), axis=3)

            loss[output_values != input_values] = 0
            loss[output_values == 0] = 0

        else:
            output_values = np.repeat(np.repeat(output_values, 2, axis=1), 2, axis=2)
            loss = np.repeat(np.repeat(loss, 2, axis=1), 2, axis=2)

            if self.del_axis_1:
                added_values = np.zeros((output_values.shape[0], 1, output_values.shape[2]))
                output_values = np.concatenate((output_values, added_values), axis=1)
                loss = np.concatenate((loss, added_values), axis=1)
            if self.del_axis_2:
                added_values = np.zeros((output_values.shape[0], output_values.shape[1], 1))
                output_values = np.concatenate((output_values, added_values), axis=2)
                loss = np.concatenate((loss, added_values), axis=2)

            loss[output_values != input_values] = 0
            loss[output_values == 0] = 0



        return loss, None
            

    def set_shape(self, new_input_shape):
        self.set_input_shape(new_input_shape)
        self.set_output_shape(np.append(new_input_shape[0], new_input_shape[1:] // 2))
        
    def control_input_and_output_shape(self):
        self.del_axis_1 = self.input_shape[1] % 2 == 1
        self.del_axis_2 = self.input_shape[2] % 2 == 1