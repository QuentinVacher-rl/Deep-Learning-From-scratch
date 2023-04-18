import numpy as np
from layers import Layer

from scipy import signal
from typing import Iterable

class Conv2D(Layer):

    def __init__(self, output_shape: Iterable, kernel=3, activation_function="relu|softmax", input_shape:Iterable=None) -> None:
        """Init Dense Layer"""
        super().__init__(output_shape, activation_function, input_shape)
        if isinstance(kernel, int) or len(kernel) == 1:
            self.kernel=np.array([kernel, kernel])
        else:
            self.kernel=np.array(kernel)
        self.input_channel = None
        self.output_channel = None
        self.mat11 = None
        self.mat22 = None


    

    def init_weight(self):
        self.weight = self.init_weight_method(np.random.random(np.concatenate(([self.output_channel, self.input_channel], self.kernel))))
        self.nb_weight = self.weight.size
        #self.init_mat_passage()
    
    def init_mat_passage(self):

        mat1 = np.array([np.concatenate((np.zeros((self.input_shape[2], i)), (np.eye(self.input_shape[2])), np.zeros((self.input_shape[2], self.kernel[1] - 1 - i))), axis=1) for i in range(self.kernel[1])])

        mat2 = np.array([np.concatenate((np.zeros((self.input_shape[1], i)), (np.eye(self.input_shape[1])), np.zeros((self.input_shape[1], self.kernel[0] - 1 - i))), axis=1) for i in range(self.kernel[0])])

        mat11 = np.concatenate([mat1[i] for i in range(self.kernel[1]) for _ in range(self.kernel[0])]).reshape(np.product(self.kernel),self.input_shape[1],self.input_shape[1]+self.kernel[1]-1)
        self.mat11 = np.concatenate([mat11 for _ in range(self.output_channel)])

        self.mat22 = np.concatenate([mat2 for _ in range(self.kernel[1])]).reshape(np.product(self.kernel),self.input_shape[2],self.input_shape[2]+self.kernel[0]-1)

        self.shape = (self.output_channel,np.product(self.kernel)*self.input_channel,self.input_shape[1],self.input_shape[1]+self.kernel[1]-1)

    def forward(self, values):


        if len(values.shape) == 4:
            # TODO change not work, CNN only for batch size = 1
            new_shape = tuple([1]) + self.weight[0].shape
            values = np.swapaxes(
                np.array([signal.convolve(values, filter[::-1,::-1,::-1].reshape(new_shape), mode="valid") for filter in self.weight]),
            0, 1).reshape(tuple([values.shape[0]]) + tuple(self.output_shape))
        else:
            values = np.array([signal.convolve(values, filter[::-1,::-1,::-1], mode="valid") for filter in self.weight]).reshape(self.output_shape)
        return self.activate_function(values)


    def backward(self, output_values, input_values, loss):
        
        error_weights = self.derivate_function(output_values.copy())*loss

        if len(output_values.shape) == 4:

            gradient = np.array([
                np.array([
                    signal.convolve(
                        input_values[:,i], error_weights[:,j,::-1,::-1],"valid") for i in range(self.input_channel)
                ]) for j in range(self.output_channel)
            ]).reshape(self.weight.shape)
            loss = np.swapaxes(
                np.sum([
                    np.array([
                        signal.convolve(
                            self.weight[j][i].reshape(1,self.kernel[0],self.kernel[1]), error_weights[:, j], mode="full")
                        for j in range(self.output_channel)
                    ]) for i in range(self.input_channel)
                ], axis=1),
            0,1)

        else:

            gradient = np.array([
                np.array([
                    signal.convolve(
                        input_values[i], error_weights[j,::-1,::-1],"valid") for i in range(self.input_channel)
                ]) for j in range(self.output_channel)
            ])
            loss = np.sum([
                np.array([
                    signal.convolve(
                        self.weight[j][i], error_weights[j], mode="full")
                    for j in range(self.output_channel)
                ]) for i in range(self.input_channel)
            ], axis=1)

        return loss, gradient

    
    def control_input_and_output_shape(self):
        """Control if the input and ouput shape respect the Conv2D layer constraints
        """

        if not isinstance(self.input_shape, Iterable)or len(self.input_shape) == 1 or len(self.input_shape) > 3:
            print("Wrong dimension of input shape for Conv2D layer (should be 2 for unbacthed and 3 for batched data)")
            raise
        elif len(self.input_shape) == 2:
            self.input_channel = 1
            self.input_shape = np.append(1, self.input_shape)
        else:
            self.input_channel = self.input_shape[0]


        if not isinstance(self.output_shape, Iterable):
            self.output_shape = np.array([
                self.output_shape,
                self.input_shape[1] - self.kernel[0] + 1,
                self.input_shape[2] - self.kernel[1] + 1
            ])
            self.output_channel = self.output_shape[0]

        elif len(self.output_shape) == 1:
            self.output_shape = np.array([
                self.output_shape[0],
                self.input_shape[1] - self.kernel[0] + 1,
                self.input_shape[2] - self.kernel[1] + 1
            ])
            self.output_channel = self.output_shape[0]
        else:
            print("Wrong dimension of output shape for Conv2D layer (should be 1)")
            raise
            

    def set_input_shape(self, new_input_shape):
        self.input_shape = new_input_shape
        
