
from typing import Iterable
from layers.layer import Layer
from optimizer import Optimizer

import numpy as np
import time 
import math


class Model:

    def __init__(self) -> None:
    
        self.layers:Iterable[Layer] = list()
        self.outputs = list()
        self.optimizer = None


    def add(self, layer: Layer):
        if not self.layers:
            if not layer.has_input_shape():
                raise "No input shape given"
        else:
            layer.set_shape(self.layers[-1].output_shape)


        layer.control_input_and_output_shape()
        layer.init_weight()

        self.layers.append(layer)

    def compile(self, optimizer="SGD", loss="MSE", lr=0.001):
        self.optimizer = Optimizer(optimizer, loss, lr, len(self.layers))
        self.optimizer.nb_weight = np.sum([layer.nb_weight for layer in self.layers])

    def test(self, l_input_values, label):

        accuracy = 0
        loss = list()
        for index, input_values in enumerate(l_input_values):

            output_values, _ = self.forward(input_values, test=True)

            accuracy += self.optimizer.calcul_accuracy(output_values, label[index])
            loss.append(self.optimizer.calcul_loss(output_values, label[index]))
        
        accuracy = round(100 * accuracy / len(l_input_values), 2)
        loss = round(np.mean(loss), 2)
        return accuracy, loss




    def fit(self, input_values, valid_output_values, nb_epoch, size_batch, validation_data:None):
        
        nb_iteration = math.trunc(input_values.shape[0] / size_batch)
        for epoch in range(nb_epoch):
            time_epoch=time.time()
            accuracy = 0
            loss = list()
            index_shuffle = np.arange(input_values.shape[0])
            np.random.shuffle(index_shuffle)
            for iteration in range(nb_iteration):
                if size_batch > 1:
                    index_used = index_shuffle[iteration:iteration+size_batch]
                else:
                    index_used = index_shuffle[iteration]

                output_values, all_values = self.forward(input_values[index_used])
                valid_output_values_it = valid_output_values[index_used]

                gradient = self.backward(all_values, valid_output_values_it) 

                accuracy += self.optimizer.calcul_accuracy(output_values, valid_output_values_it)
                loss.append(self.optimizer.calcul_loss(output_values, valid_output_values_it))
                
                self.apply_gradient(gradient)
                if iteration > 0:
                    print(f"epoch {iteration} --- Time : {round(time.time()-time_epoch)}s --- Loss : {round(np.mean(loss), 2)} --- Accuracy : {round(100 * accuracy/(iteration*size_batch), 2)}%", end="\r")

            if not validation_data is None:
                val_accuracy, val_loss = self.test(validation_data[0], validation_data[1])

                
            print(f"epoch {epoch} --- Time : {round(time.time()-time_epoch)}s --- Loss : {round(np.mean(loss), 2)} --- Accuracy : {round(100 * accuracy/len(input_values), 2)}% --- Val Loss : {val_loss} --- Val Accuracy : {val_accuracy}")
                    


    def forward(self, input_values, test=False):
        all_values = [input_values]
        values = input_values
        for layer in self.layers:
            if test and layer.desactivated_during_test:
                continue
            values = layer.forward(values)
            all_values.append(values)
        return values, all_values


    def backward(self, values, expected_values):
        return self.collect_gradient(values, expected_values)

    def collect_gradient(self, values, expected_values):
        gradient = list()
        loss = self.optimizer.calcul_derivate_loss(values[-1], expected_values) 
        for index in reversed(range(len(self.layers))):
            loss, layer_grad = self.layers[index].backward(values[index+1].copy(), values[index].copy(), loss)
            if not layer_grad is None: 
                gradient.insert(0, layer_grad)
        return np.array(gradient, dtype=np.ndarray)

    def collect_weights(self):
        return np.concatenate([layer.weight.reshape(-1) for layer in self.layers])

    def apply_gradient(self, gradient):
        gradient = self.optimizer.calcul_optim_gradient(gradient)
        index = 0
        for layers in self.layers:
            if layers.trainable:
                layers.weight -= gradient[index]
                index += 1
            

    def predict(self):
        pass