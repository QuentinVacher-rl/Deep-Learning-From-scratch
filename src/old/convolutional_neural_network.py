
import numpy as np
from scipy import signal

import math
import time

class CNN:

    def __init__(self, p_input_size: int, p_output_size: int, p_hidden_layer: list, p_activation_function="ReLu", p_optimizer="Adam", p_size_filter=[3,3], p_nb_filter=1) -> None:
        
        self.input_shape = (28,28)

        self.input_size = p_input_size

        self.output_size = p_output_size

        self.hidden_layer = p_hidden_layer

        self.l_layer = np.array([self.input_size] + self.hidden_layer + [self.output_size])

        self.activation_function = p_activation_function

        self.l_bias = list()
        self.l_weight = list()

        self.all_weight = np.array
        
        self.lr = 0.01
        self.init_lr = self.lr
        self.final_lr = 0.0001

        self.optimizer = p_optimizer

        for index in range(1, len(self.l_layer)):
            self.l_weight.append((np.random.rand(self.l_layer[index-1],self.l_layer[index]).transpose()*2-1))
            self.l_bias.append((np.random.rand(self.l_layer[index])*2-1))
        
        self.nb_layer = 2 + len(self.hidden_layer)
        


        # CNN parameters
        self.size_filter = p_size_filter
        self.nb_filter = p_nb_filter


        self.l_filters = list()
        dim_image = 1
        self.l_size = [dim_image,4,8,16]
        for index, size in enumerate(self.l_size[1:]):
            self.l_filters.append(np.random.rand(size, self.l_size[index], self.size_filter[0], self.size_filter[0]))

        self.all_values_ff = list()
        self.all_values_cnn = list()

        self.pooling_saved_value = list()

        self.histo_shape = np.array([[1,28,28], [4,26,26], [4,13,13], [8,11,11], [8,5,5], [16,3,3]])


        # Adam parameters
        self.t=0
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 10**(-8)
        self.adam_w1 = [0]*(self.nb_layer-1)
        self.adam_b1 = [0]*(self.nb_layer-1)
        self.adam_w2 = [0]*(self.nb_layer-1)
        self.adam_b2 = [0]*(self.nb_layer-1)
        self.adam_f1 = [0]*(len(self.l_filters))
        self.adam_f2 = [0]*(len(self.l_filters))

    def test(self, input_values, label):

        accuracy = 0
        for index, input_value in enumerate(input_values):
            expected_output = np.zeros(self.output_size)
            expected_output[int(label[index])] = 1

            output_value, _ = self.feed_forward(input_value)

            accuracy += 1 if int(label[index]) == np.argmax(output_value) else 0
        
        accuracy = round(100 * accuracy / len(input_values), 2)
        return accuracy


    def fit(self, input_values, valid_output_values, nb_epoch, size_batch):
        
        nb_iteration = math.trunc(input_values.shape[0] / size_batch)
        info_time = {
            "f": 0,
            "f_nn": 0,
            "f_c": 0,
            "f_p": 0,
            "b": 0,
            "b_nn": 0,
            "b_c": 0,
            "b_p": 0,

        }
        for epoch in range(nb_epoch):
            self.lr *= 0.99
            accuracy = 0
            l_loss = list()
            index_shuffle = np.arange(input_values.shape[0])
            np.random.shuffle(index_shuffle)
            index = 0
            for iteration in range(nb_iteration):
                global_gradient_weight = list()
                global_gradient_bias = list()
                global_gradient_filter = list()

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
                        global_gradient_filter = result["filter"]
                    else:
                        for layer in range(self.nb_layer-1):
                            global_gradient_weight[layer] += result["weight"][layer]
                            global_gradient_bias[layer] += result["bias"][layer]
                        for layer_filter in range(len(self.l_filters)):
                            global_gradient_filter[layer_filter] += result["filter"][layer_filter]

                    time = result["time"]
                    info_time["f"] += time["forward"]
                    info_time["f_nn"] += time["det_forward"]["nn"]
                    info_time["f_c"] += time["det_forward"]["conv"]
                    info_time["f_p"] += time["det_forward"]["pool"]
                    info_time["b"] += time["backward"]
                    info_time["b_nn"] += time["det_backward"]["nn"]
                    info_time["b_c"] += time["det_backward"]["conv"]
                    info_time["b_p"] += time["det_backward"]["pool"]
                
                    l_loss.append(result["loss"])
                    accuracy += result["accuracy"]



                index += size_batch

                global_gradient_weight.reverse()
                global_gradient_bias.reverse()
                global_gradient_filter.reverse()
                if self.optimizer == "SGD":
                    for layer in (range(self.nb_layer-1)):
                        self.l_weight[layer] -= self.lr * global_gradient_weight[layer]
                        self.l_bias[layer] -= self.lr * global_gradient_bias[layer]
                        
                    
                    for layer_filter in range(len(self.l_filters)):
                        self.l_filters[layer_filter] -= self.lr * global_gradient_filter[layer_filter]

                elif self.optimizer == "Adam":
                    self.t += 1
                    for layer in (range(self.nb_layer-1)):
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
                    
                    for layer_filter in range(len(self.l_filters)):
                        self.adam_f1[layer_filter] = self.b1*self.adam_f1[layer_filter] + (1-self.b1) * global_gradient_filter[layer_filter]
                        f1_corrected = self.adam_f1[layer_filter] / (1 - np.power(self.b1, self.t))

                        self.adam_f2[layer_filter] = self.b2*self.adam_f2[layer_filter] + (1-self.b2) * np.power(global_gradient_filter[layer_filter], 2)
                        f2_corrected = self.adam_f2[layer_filter] / (1 - np.power(self.b2, self.t))

                        self.l_filters[layer_filter] = self.l_filters[layer_filter] - self.lr * f1_corrected / (np.sqrt(f2_corrected)+self.epsilon)

                print(f"iteration {iteration} --- mean loss : {round(np.mean(l_loss), 2)} --- accuracy : {round(100 * accuracy/(size_batch*(1+iteration)), 2)}%")
                print(f"Forward : {info_time['f']} : nn : {info_time['f_nn']}, conv : {info_time['f_c']}, pool : {info_time['f_p']}")
                print(f"Backward : {info_time['b']} : nn : {info_time['b_nn']}, conv : {info_time['b_c']}, pool : {info_time['b_p']}")
            print(f"epoch {epoch} --- mean loss : {round(np.mean(l_loss), 2)} --- accuracy : {round(100 * accuracy/len(input_values), 2)}%")
                    
                
    def collect_gradient_for_mp(self, data_mp):
        input_value = data_mp["input"]
        valid_output_value = data_mp["output"]
        expected_output = np.zeros(self.output_size)
        expected_output[int(valid_output_value)] = 1

        # Forward
        time_1 = time.time()
        output_value, time_forward_detailed = self.feed_forward(input_value)
        time_2 = time.time()

        # Backward
        gradient_weight, gradient_bias, gradient_filter, _, time_backward_detailed = self.get_descent_gradiant(expected_output)
        time_3 = time.time()

        #save metrics
        loss = np.sum((expected_output - output_value)**2)
        accuracy = 1 if int(valid_output_value) == np.argmax(output_value) else 0
        

        time_forward = time_2 - time_1
        time_backward = time_3 - time_2
        info_time = {
            "forward": time_forward,
            "backward": time_backward,
            "det_forward": time_forward_detailed,
            "det_backward": time_backward_detailed
        }
        return {
            "weight": gradient_weight,
            "bias": gradient_bias,
            "filter": gradient_filter,
            "loss": loss,
            "accuracy": accuracy,
            "time": info_time
        }


    def feed_forward(self, input_values):


        self.all_values_cnn.clear()
        self.all_values_ff.clear()

        values = input_values.reshape(1,28,28)

        self.all_values_cnn.append(values)
        t2=time.time()
        next_values = self.conv_forward(values, 0)
        t3=time.time()
        values = self.maxpooling_forward(next_values)
        t4=time.time()
        next_values = self.conv_forward(values, 1)
        t5=time.time()
        values = self.maxpooling_forward(next_values)
        t6=time.time()
        next_values = self.conv_forward(values, 2)
        t7=time.time()
        
        #flatten
        input_values = next_values.reshape(next_values.size)

        #Normalisation:
        input_values = self.activate_function(input_values)



        #values = (input_values - np.min(input_values))/(np.max(input_values)-np.min(input_values))
        values = input_values
        
        for layer in range(self.nb_layer-1):
            self.all_values_ff.append(values)
            values = self.l_weight[layer] @ values + self.l_bias[layer]
            values = self.activate_function(values.copy())


        self.all_values_ff.append(values)
        t8 = time.time()
        time_detailed = {
            "nn":t8-t7,
            "conv":t3-t2 + t5-t4 + t7-t6,
            "pool": t4-t3 + t6-t5
        }
        return values, time_detailed
        

    def conv_forward(self, values, step):
        if True:

            next_values = np.array([signal.convolve(values, filter, mode="valid") for filter in self.l_filters[step]]).reshape(self.l_filters[step].shape[0], values.shape[1]-2, values.shape[2]-2)
       
        # Relu activation
        next_values = self.activate_function(next_values, "leaky ReLu")
        self.all_values_cnn.append(next_values)
        return next_values
                    
    def maxpooling_forward(self, input_values):
        # Delete row non multiple of 2
        if input_values.shape[1] % 2 == 1:
            input_values = np.delete(input_values, (-1), axis=1)
        if input_values.shape[2] % 2 == 1:
            input_values = np.delete(input_values, (-1), axis=2)

        nb_filter = input_values.shape[0]
        shape_0 = int(input_values.shape[1]/2)
        shape_1 = int(input_values.shape[2]/2)
        tuple_value_sup = tuple(sorted([i+shape_1 + 2 * j * shape_1 for i in range(int(shape_1)) for j in range(shape_0)]))

        reshape_values = input_values.reshape(nb_filter,(shape_0*shape_1*2),2)
        reshape_values_r = np.roll(reshape_values, -shape_1, axis=1)

        next_values = np.delete(np.concatenate((reshape_values, reshape_values_r), axis=2), tuple_value_sup, axis=1)
        max_next_values = np.max(next_values, axis=2).reshape(nb_filter,shape_0,shape_1)
        
        self.pooling_saved_value.append(next_values)

        self.all_values_cnn.append(max_next_values)

        return max_next_values


    def feed_backward(self, expected_output):
        gradient_weight, gradient_bias, _ = self.get_descent_gradiant(expected_output)

        for layer in (range(1, self.nb_layer)):
            self.l_weight[layer-1] -= gradient_weight[self.nb_layer - layer - 1]
            self.l_bias[layer-1] -= gradient_bias[self.nb_layer - layer - 1]

    def get_descent_gradiant(self, expected_output):
        
        t1 = time.time()
        gradient_weight = list()
        gradient_bias = list()
        loss = loss = (self.all_values_ff[-1].copy() - expected_output) * 2 / expected_output.shape[0]
        for layer in reversed((range(1, self.nb_layer))):
            values_out = self.all_values_ff[layer].copy()
            values_in_layer_before = self.all_values_ff[layer-1]
            dout_din = self.derivate_function(values_out)

            gradient_weight.append(np.tensordot(dout_din*loss, values_in_layer_before, 0))
            gradient_bias.append(dout_din*loss)

            loss = self.l_weight[layer-1].T @ (loss*dout_din)
        t2 = time.time()

        gradient_filter = list()

        t3 = time.time()
        f_delta_grad, new_loss = self.conv_backward(loss.reshape(self.histo_shape[-1]), 0, 0)
        gradient_filter.append(f_delta_grad)

        t4 = time.time()
        new_loss = self.maxpooling_backward(new_loss, 1, 0)

        t5 = time.time()
        f_delta_grad, new_loss = self.conv_backward(new_loss, 2, 1)
        gradient_filter.append(f_delta_grad)

        t6 = time.time()
        new_loss = self.maxpooling_backward(new_loss, 3, 1)

        t7 = time.time()
        f_delta_grad, new_loss = self.conv_backward(new_loss, 4, 2)
        gradient_filter.append(f_delta_grad)
        t8 = time.time()

        time_detailed = {
            "nn":t2-t1,
            "conv":t8-t7 + t6-t5 + t4-t3,
            "pool":t7-t6 + t5-t4
        }
        return gradient_weight, gradient_bias, gradient_filter, loss, time_detailed

    def conv_backward(self, loss, step, conv_step):
        f = self.l_filters[-(conv_step+1)]
        x = self.all_values_cnn[-(step+2)]
        f_delta_grad = np.zeros(f.shape)
        new_loss = np.zeros(self.histo_shape[-(step+2)])
        if False:
            for j in range(f.shape[0]):
                for i in range(x.shape[0]):
                    f_delta_grad[j][i] = signal.convolve(x[i],loss[j],"valid")
                    new_loss[i] += signal.convolve(np.rot90(np.rot90(f[j][i])), loss[j], mode="full")

        if True:
            f_delta_grad = np.array([np.array([signal.convolve(x[i], loss[j],"valid") for i in range(x.shape[0])]) for j in range(f.shape[0])]).reshape(f.shape)
            new_loss = np.sum([np.array([signal.convolve(np.rot90(np.rot90(f[j][i])), loss[j], mode="full") for j in range(f.shape[0])]) for i in range(x.shape[0])], axis=1)

        new_loss = new_loss * self.derivate_function(new_loss, "leaky ReLu")
        
        return f_delta_grad, new_loss
                

    def maxpooling_backward(self, loss, step, pool_step):

        shape = self.histo_shape[-(step+1)]
        nb_filter = shape[0]
        shape_0 = int(shape[1])
        shape_1 = int(shape[2])
        tuple_value_sup = tuple(sorted([2+i + 2 * j * shape_1 for i in range(int((shape_1-1)*2)) for j in range(shape_0)]))

        values = self.pooling_saved_value[-(pool_step+1)]

        values = values.reshape(int(values.shape[0]*values.shape[1]*values.shape[2]/4), 4)
        filter = (values.T / (values.max(axis=1)-self.epsilon)).T < 1
        values[filter] = 0
        
        values[filter == False] = 1
        values = (values.T * loss.reshape(loss.size)).T


        h = values.reshape(nb_filter,(shape_0*shape_1*2),2)
        j = np.concatenate([np.roll(h, -i*2, axis=1) for i in range(shape_1)], axis=2)
        new_a = np.delete(j, tuple_value_sup, axis=1).reshape(nb_filter, shape_0*2, shape_1*2)

        if shape_1*2 < self.histo_shape[-(step+2)][2]:
            new_a = np.concatenate((new_a, np.zeros((new_a.shape[0],new_a.shape[1],1))), axis=2)
        if shape_0*2 < self.histo_shape[-(step+2)][1]:
            new_a = np.concatenate((new_a, np.zeros((new_a.shape[0],1,new_a.shape[2]))), axis=1)

        return new_a

    def activate_function(self, values, activation_function=None):
        if activation_function is None:
            activation_function = self.activation_function
        if activation_function == "ReLu":
            values[values<0] = 0
            return values

        if activation_function == "leaky ReLu":
            values = np.maximum(values, 0.01 * values)
            return values

        if activation_function == "Sigmoid":
            values = 1 / (1 + np.exp(-1 * values))
            return values
        else:
            raise NameError(f"Wrong activation function given : {str(activation_function)}, try between 'ReLu', 'leaky ReLu' and 'Sigmoid'")

    def derivate_function(self, values, activation_function=None):
        if activation_function is None:
            activation_function = self.activation_function

        if activation_function == "ReLu":
            values[values<0] = 0
            values[values>0] = 1
            return values

        if activation_function == "leaky ReLu":
            values[values<0] = 0.01
            values[values>0] = 1
            return values


        if activation_function == "Sigmoid":
            values = values * (1-values)
            return values
        else:
            raise NameError(f"Wrong activation function given : {str(activation_function)}, try between 'ReLu', 'leaky ReLu' and 'Sigmoid'")