
import numpy as np

import math
import time


class Neural_Network:

    def __init__(self, p_input_size: int, p_output_size: int, p_hidden_layer: list, p_activation_function="ReLu", p_optimizer="Adam") -> None:
        
        self.input_size = p_input_size

        self.output_size = p_output_size

        self.hidden_layer = p_hidden_layer

        self.l_layer = np.array([self.input_size] + self.hidden_layer + [self.output_size])

        self.activation_fuction = p_activation_function

        self.l_bias = list()
        self.l_weight = list()

        self.all_weight = np.array
        
        self.lr = 0.001
        self.init_lr = self.lr
        self.final_lr = 0.0001

        self.optimizer = p_optimizer

        for index in range(1, len(self.l_layer)):
            self.l_weight.append((np.random.rand(self.l_layer[index-1],self.l_layer[index]).transpose()*2-1))
            self.l_bias.append((np.random.rand(self.l_layer[index])*2-1))
        
        self.nb_layer = 2 + len(self.hidden_layer)
        
        self.dropout = False
        self.dropout_coef = 0.1

        self.t=0
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 10**(-8)
        self.adam_w1 = [0]*(self.nb_layer-1)
        self.adam_b1 = [0]*(self.nb_layer-1)
        self.adam_w2 = [0]*(self.nb_layer-1)
        self.adam_b2 = [0]*(self.nb_layer-1)

    def test(self, input_values, label):

        accuracy = 0
        loss = list()
        for index, input_value in enumerate(input_values):
            expected_output = np.zeros(self.output_size)
            expected_output[int(label[index])] = 1

            output_value, _, _ = self.feed_forward(input_value, test=True)

            accuracy += 1 if int(label[index]) == np.argmax(output_value) else 0
            loss.append(np.sum((expected_output - output_value)**2))
        
        accuracy = round(100 * accuracy / len(input_values), 2)
        loss = round(np.mean(loss), 2)
        return accuracy, loss


    def fit(self, input_values, valid_output_values, nb_epoch, size_batch, validation_data:None):
        
        nb_iteration = math.trunc(input_values.shape[0] / size_batch)
        for epoch in range(nb_epoch):
            time_epoch=time.time()
            self.lr *= 0.99
            accuracy = 0
            l_loss = list()
            index_shuffle = np.arange(input_values.shape[0])
            np.random.shuffle(index_shuffle)
            index = 0
            for iteration in range(nb_iteration):
                global_gradient_weight = list()
                global_gradient_bias = list()

                data_mp = list()
                for index_mp in range(size_batch):
                    index_used = index_shuffle[index_mp]
                    data_mp.append({
                        "input": input_values[index_used],
                        "output": valid_output_values[index_used]
                    })
                

                results_mp = list()
                for data in data_mp:
                    results_mp.append(self.collect_gradient_for_mp(data))

                for index_result, result in enumerate(results_mp):
                    if index_result == 0:
                        global_gradient_weight = result["weight"]
                        global_gradient_bias = result["bias"]
                    else:
                        for layer in range(self.nb_layer-1):
                            global_gradient_weight[layer] += result["weight"][layer]
                            global_gradient_bias[layer] += result["bias"][layer]
                
                    l_loss.append(result["loss"])
                    accuracy += result["accuracy"]



                index += size_batch

                global_gradient_weight.reverse()
                global_gradient_bias.reverse()
                if self.optimizer == "SGD":
                    for layer in (range(0, self.nb_layer-1)):
                        self.l_weight[layer] -= self.lr * global_gradient_weight[layer]
                        self.l_bias[layer] -= self.lr * global_gradient_bias[layer]
                elif self.optimizer == "Adam":
                    self.t += 1
                    for layer in (range(0, self.nb_layer-1)):
                        self.adam_w1[layer] = self.b1*self.adam_w1[layer] + (1-self.b1) * global_gradient_weight[layer]
                        w1_corrected = self.adam_w1[layer] / (1 - np.power(self.b1, self.t))

                        self.adam_b1[layer] = self.b1*self.adam_b1[layer] + (1-self.b1) * global_gradient_bias[layer]
                        b1_corrected = self.adam_b1[layer] / (1 - np.power(self.b1, self.t))


                        self.adam_w2[layer] = self.b2*self.adam_w2[layer] + (1-self.b2) * np.power(global_gradient_weight[layer], 2)
                        w2_corrected = self.adam_w2[layer] / (1 - np.power(self.b2, self.t))

                        self.adam_b2[layer] = self.b2*self.adam_b2[layer] + (1-self.b2) * np.power(global_gradient_bias[layer], 2)
                        b2_corrected = self.adam_b2[layer] / (1 - np.power(self.b2, self.t))
                    
                        self.l_weight[layer] = self.l_weight[layer] - self.lr * w1_corrected / (np.sqrt(w2_corrected)+self.epsilon)
                        self.l_bias[layer] = self.l_bias[layer] - self.lr * b1_corrected / (np.sqrt(b2_corrected)+self.epsilon)

            val_str = ""
            if not validation_data is None:
                val_accuracy, val_loss = self.test(validation_data[0], validation_data[1])
                val_str = f" --- Val loss"


            print(f"epoch {epoch} --- Time : {round(time.time()-time_epoch)}s --- Loss : {round(np.mean(l_loss), 2)} --- Accuracy : {round(100 * accuracy/len(input_values), 2)}% --- Val Loss : {val_loss} --- Val Accuracy : {val_accuracy}")
                    
                
    def collect_gradient_for_mp(self, data_mp):
        input_value = data_mp["input"]
        valid_output_value = data_mp["output"]
        expected_output = np.zeros(self.output_size)
        expected_output[int(valid_output_value)] = 1

        # Forward
        output_value, all_neuron_values, dropout_filter = self.feed_forward(input_value)


        # Backward
        gradient_weight, gradient_bias = self.get_descent_gradiant(all_neuron_values, expected_output, dropout_filter)



        #save metrics
        loss = np.sum((expected_output - output_value)**2)
        accuracy = 1 if int(valid_output_value) == np.argmax(output_value) else 0
        
        return {
            "weight": gradient_weight,
            "bias": gradient_bias,
            "loss": loss,
            "accuracy": accuracy
        }


    def feed_forward(self, input_values, test=False):

        all_values = list()
        dropout_filter = list()

        #values = (input_values - np.min(input_values))/(np.max(input_values)-np.min(input_values))
        values = input_values
        
        
        for layer in range(self.nb_layer-1):
            all_values.append(values)

            weight = self.l_weight[layer].copy()
            bias = self.l_bias[layer].copy()
            if self.dropout and not test:
                dropout_filter.append({
                    "weight":np.random.random(weight.shape), 
                    "bias":np.random.random(bias.shape)
                })
                weight *= 1 / (1-self.dropout_coef)
                bias *= 1 / (1-self.dropout_coef)

                weight[dropout_filter[layer]["weight"]<self.dropout_coef] = 0
                bias[dropout_filter[layer]["bias"]<self.dropout_coef] = 0

            

            values = weight @ values + bias
            values = self.activate_function(values.copy())


        all_values.append(values)
        return values, all_values, dropout_filter
        
    def get_descent_gradiant(self, all_values, expected_output, dropout_filter):
        
        gradient_weight = list()
        gradient_bias = list()
        loss = (expected_output - all_values[-1].copy()) * 2 / expected_output.shape[0]
        for layer in reversed((range(1, self.nb_layer))):
            values_out = all_values[layer].copy()
            values_in_layer_before = all_values[layer-1]
            dout_din = self.derivate_function(values_out)

            g_weight = np.tensordot(dout_din*loss, values_in_layer_before, 0)
            g_bias = dout_din*loss

            weight = self.l_weight[layer-1].copy()

            gradient_weight.append(g_weight)
            gradient_bias.append(g_bias)


            
            loss = weight.T @ (loss*dout_din)
        return gradient_weight, gradient_bias


    def feed_backward(self, all_values, expected_output):
        gradient_weight, gradient_bias = self.get_descent_gradiant(all_values, expected_output)

        for layer in (range(1, self.nb_layer)):
            self.l_weight[layer-1] -= gradient_weight[self.nb_layer - layer - 1]
            self.l_bias[layer-1] -= gradient_bias[self.nb_layer - layer - 1]


    def activate_function(self, values):
        if self.activation_fuction == "ReLu":
            values[values<0] = 0
            return values

        if self.activation_fuction == "leaky ReLu":
            values = np.max(values, 0.01 * values)
            return values

        if self.activation_fuction == "Sigmoid":
            values = 1 / (1 + np.exp(-1 * values))
            return values

    def derivate_function(self, values):
        if self.activation_fuction == "ReLu":
            values[values<0] = 0
            values[values>0] = 1
            return values

        if self.activation_fuction == "leaky ReLu":
            values[values<0] = 0.01
            values[values>0] = 1
            return values


        if self.activation_fuction == "Sigmoid":
            values = values * (1-values)
            return values


       