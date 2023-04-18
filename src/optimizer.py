from typing import Iterable
import numpy as np 

class Optimizer:

    def __init__(self, optimizer="SGD", loss="MSE", lr=0.001, nb_layer=0) -> None:
        self.optimizer = optimizer
        self.loss = loss

        self.lr = lr

        self.nb_weight = 0

        if self.optimizer == "adam":
            self.t=0
            self.b1 = 0.9
            self.b2 = 0.99
            self.epsilon = 10**(-8)
            self.adam_vect1 = np.zeros(1)
            self.adam_vect2 = np.zeros(1)

    
    def calcul_optim_gradient(self, gradient):
        if self.optimizer == "SGD":
            return self.sgd_optimizer(gradient)
        if self.optimizer == "adam":
            return self.adam_optimizer(gradient)


    def sgd_optimizer(self, gradient):
        return gradient * self.lr

    def adam_optimizer(self, gradient):
        self.t += 1

        self.adam_vect1 = self.b1*self.adam_vect1 + (1-self.b1) * gradient
        w1_corrected = self.adam_vect1 / (1 - np.power(self.b1, self.t))
        
        self.adam_vect2 = self.b2*self.adam_vect2 + (1-self.b2) * np.power(gradient, 2)
        w2_corrected = self.adam_vect2 / (1 - np.power(self.b2, self.t))

        return self.lr * w1_corrected / (np.power(w2_corrected, 1/2) + self.epsilon) 


    def adam_calcul_gradient(self, gradient, step):

        self.adam_vect1[step] = self.b1*self.adam_vect1[step] + (1-self.b1) * gradient
        w1_corrected = self.adam_vect1[step] / (1 - np.power(self.b1, self.t))
        
        self.adam_vect2[step] = self.b2*self.adam_vect2[step] + (1-self.b2) * np.power(gradient, 2)
        w2_corrected = self.adam_vect2[step] / (1 - np.power(self.b2, self.t))

        return w1_corrected / (np.sqrt(w2_corrected)+self.epsilon)

    def calcul_accuracy(self, values, expected_values):
        axis_argmax = len(values.shape) - 1
        if len(expected_values.shape) == len(values.shape):
            expected_values = np.argmax(expected_values, axis=axis_argmax)
        return np.sum([expected_values == np.argmax(values, axis=axis_argmax)])

    def calcul_loss(self, values, expected_values):
        if not isinstance(expected_values, Iterable):
            expected_values = self.get_vector_output_1D(values, expected_values)
        elif len(values.shape) == 2 and len(expected_values.shape) == 1:
            expected_values = self.get_vector_output_2D(values, expected_values)
        return np.mean(np.sum((expected_values - values)**2, axis=len(values.shape)-1))


    def calcul_derivate_loss(self, values, expected_values):
        if not isinstance(expected_values, Iterable):
            expected_values = self.get_vector_output_1D(values, expected_values)
        elif len(values.shape) == 2 and len(expected_values.shape) == 1:
            expected_values = self.get_vector_output_2D(values, expected_values)
        return (values - expected_values)


    def get_vector_output_1D(self, values, expected_values):
        expected_output_vect = np.zeros(values.shape)
        expected_output_vect[expected_values] = 1
        return expected_output_vect


    def get_vector_output_2D(self, values, expected_values):
        expected_output_vect = np.zeros(values.shape)
        np.put_along_axis(expected_output_vect, expected_values.reshape(-1,1), 1, axis=1)
        return expected_output_vect