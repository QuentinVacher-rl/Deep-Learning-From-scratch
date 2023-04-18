import math
import numpy as np
from layers.layer import Layer

from typing import Iterable

class Dense(Layer):

    def __init__(self, output_shape: Iterable, activation_function="relu|softmax", input_shape:Iterable=None) -> None:
        """Init Dense Layer"""
        super().__init__(output_shape, activation_function, input_shape)



    

    def init_weight(self):
        self.weight = self.init_weight_method((np.random.rand(self.input_shape+1,self.output_shape).transpose()))
        self.nb_weight = self.weight.size

    def forward(self, values):
        if len(values.shape) == 2:
            values = (self.weight @ np.concatenate((values, np.ones((values.shape[0],1))), axis=1).T).T
        else:
            values = self.weight @ np.append(values,1)
        return self.activate_function(values)


    def backward(self, output_values, input_values, loss):
        
        error_weights = self.derivate_function(output_values.copy())*loss
        if len(input_values.shape) == 2:
            gradient = np.tensordot(error_weights.T, np.concatenate((input_values,np.ones((input_values.shape[0],1))), axis=1), 1)
        else:
            gradient = np.tensordot(error_weights, np.append(input_values,1), 0)
        
        return np.delete((self.weight.T @ error_weights.T).T, -1, axis=len(input_values.shape)-1), gradient

    
    def control_input_and_output_shape(self):
        """Control if the input and ouput shape respect the Dense layer constraints
        """

        if not isinstance(self.input_shape, Iterable):
            pass
        elif len(self.input_shape) == 1:
            self.input_shape = self.input_shape[0]
        else:
            print("Wrong dimension of input shape for Dense layer (should be 1 for unbacthed and 2 for batched data)")
            raise

        if not isinstance(self.output_shape, Iterable):
            pass
        elif len(self.output_shape) == 1:
            self.output_shape = self.output_shape[0]
        else:
            print("Wrong dimension of output shape for Dense layer (should be 1)")
            raise


        
