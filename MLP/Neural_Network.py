#!/usr/bin/env python

import random
import math
import pickle

"""
Neural_Netwotk.py

Creates a data structure that models artificial neural network

Authors:
    Joseph Fuchs <jjf2614@rit.edu>
    Damien Cremilleux <dxc9849@rit.edu>
"""

def sigmoid(t):
    '''
    The sigmoid function
    '''
    return 1 / (1 + math.exp(-t))

def derivative_of_sigmoid(t):
    return sigmoid(t) * (1 - sigmoid(t))

class Neuron:
    '''
    Models a neuron
    '''

    nb_neurons = 0

    def __init__(self):
        '''Initilizes a neuron'''
        self.value = None
        self.id = Neuron.nb_neurons
        Neuron.nb_neurons = Neuron.nb_neurons + 1
        self.is_bias = False    # By default, neurons are not bias

    def set_value(self, value):
        '''Sets the value of the neurons'''
        if self.is_bias:
            self.value = 1      # The value of a bias node is always one
        else:
            self.value = value

    def get_value(self):
        '''Returns the value of the neurons'''
        return self.value

    def set_inj(self, inj):
        '''Sets the inj (sum of predecessors weighted) of the neurons'''
        self.inj = inj

    def get_inj(self):
        '''Returns the inj of the neurons'''
        return self.inj

    def get_id(self):
        '''Returns the id of the neurons'''
        return self.id

    def set_bias(self):
        '''Transform the neuron into a bias one'''
        self.is_bias = True
        self.set_value(1)

    @staticmethod
    def reset_nb_neurons():
        '''Reset the counter of neurons'''
        Neuron.nb_neurons = 0

    def __str__(self):
        '''Print function for a neuron'''
        if self.is_bias:
            return "Neuron (Bias) " + str(self.id) + ", value " + str(self.value)
        else:
            return "Neuron " + str(self.id) + ", value " + str(self.value)



class MLP:
    ''' Multi-layer perceptron class. '''

    learning_rate = 0.1

    def __init__(self, nb_input_nodes, nb_hidden_nodes, nb_output_nodes):
        ''' Initialization of the perceptron with given sizes.'''

        self.nb_nodes = nb_input_nodes + nb_hidden_nodes + nb_output_nodes + 2 # We add bias nodes

        # We initializes the neurons
        Neuron.reset_nb_neurons()
        self.neurons = []
        input_layer = []
        for ii in range(0, nb_input_nodes + 1):
            input_layer.append(Neuron())
        input_layer[0].set_bias()
        self.neurons.append(input_layer)

        hidden_layer = []
        for ii in range(0, nb_hidden_nodes + 1):
            hidden_layer.append(Neuron())
        hidden_layer[0].set_bias()
        self.neurons.append(hidden_layer)

        output_layer = []
        for ii in range(0, nb_output_nodes):
            output_layer.append(Neuron())
        self.neurons.append(output_layer)

        self.weight = []        # Matrice with the weight, some values will be irrelevant (eg weight[1][1])

        # Initialize with random values in [-1,1]
        random.random()
        for ii in range(0, self.nb_nodes):
            layer = []
            for jj in range(0, self.nb_nodes):
                w = random.uniform(-1,1)
                layer.append(w)
            self.weight.append(layer)

        # Initialize epoch
        self.epoch = 0

    def get_weight(self):
        '''Return the matrice of weight'''
        return self.weight

    def get_input_layer(self):
        '''Return the input layer of neurons'''
        return self.neurons[0]

    def get_hidden_layer(self):
        ''' Return the hidden layer of neurons'''
        return self.neurons[1]

    def get_output_layer(self):
        '''Return the output layer of neurons'''
        return self.neurons[2]

    def get_neuron(self, i, j):
        '''Return the j neuron of the i layer'''
        return self.neurons[i][j]

    def inc_epoch(self):
        '''Increment the epoch'''
        self.epoch = self.epoch +1

    def get_epoch(self):
        '''Return the epoch'''
        return self.epoch

    def propagate_inputs(self, l_input):
        ''' Propagate the inputs forward to compute the ouput '''

        # Set input layer
        for ii in range(0,len(l_input)):
                self.get_neuron(0,ii+1).set_value(l_input[ii])

        # Propagate from input_layer to hidden_layer
        for nh in self.get_hidden_layer():
            inj = 0
            for ni in self.get_input_layer():
                inj = inj + self.weight[ni.get_id()][nh.get_id()] * ni.get_value()
            nh.set_inj(inj)
            nh.set_value(sigmoid(inj))

        # Propagate from hidden_layer to ouput_layer
        for no in self.get_output_layer():
            inj = 0
            for nh in self.get_hidden_layer():
                inj = inj + self.weight[nh.get_id()][no.get_id()] * nh.get_value()
            no.set_inj(inj)
            no.set_value(sigmoid(inj))

        # Return the values of the ouputs
        ouputs = []
        for no in self.get_output_layer():
            ouputs.append(no.get_value())
        return ouputs


    def propagate_outputs(self, l_output):
        '''Back propagation learning'''

        delta = []              # a vector of errors, indexed by network node
        for ii in range(0, self.nb_nodes):
            delta.append(None)

        # Propagate delta backward from output layer to input layer
        # Output layer
        for ii in range(0, len(self.get_output_layer())):
            delta[self.get_output_layer()[ii].get_id()] = (l_output[ii] - self.get_output_layer()[ii].get_value()) * derivative_of_sigmoid(self.get_output_layer()[ii].get_inj())

        # Hidden layer
        for nh in self.get_hidden_layer():
            sum_tmp = 0
            for no in self.get_output_layer():
                sum_tmp = sum_tmp + self.weight[nh.get_id()][no.get_id()] * delta[no.get_id()]
            delta[nh.get_id()] =  derivative_of_sigmoid(nh.get_inj()) * sum_tmp

        # Input layer
        for ni in self.get_input_layer():
            sum_tmp = 0
            for nh in self.get_hidden_layer():
                sum_tmp = sum_tmp + self.weight[ni.get_id()][nh.get_id()] * delta[nh.get_id()]
            delta[ni.get_id()] =  ni.get_value() *  (1 - ni.get_value()) * sum_tmp

        # Update every weight in network using delta
        for ii in range(0, self.nb_nodes):
            if ii < len(self.get_input_layer()):
                n = self.get_input_layer()[ii]
            elif ii < (len(self.get_input_layer()) + len(self.get_hidden_layer())):
                n = self.get_hidden_layer()[ii - len(self.get_input_layer())]
            else:
                n =  self.get_output_layer()[ii - len(self.get_input_layer()) - len(self.get_hidden_layer())]
            for jj in range(0, self.nb_nodes):
                self.weight[ii][jj] = self.weight[ii][jj] + MLP.learning_rate * n.get_value() * delta[jj]

    def errors(self, l_output):
        '''Return the sum of square errors'''
        sum = 0
        for ii in range(0, len(self.get_output_layer())):
            sum = sum + (self.get_output_layer()[ii].get_value() - l_output[ii]) ** 2
        return sum

    def train_MLP(self, l_input, l_output):
        '''Train the MLP with the given input/ouput and return the errors'''
        self.propagate_inputs(l_input)
        self.propagate_outputs(l_output)
        return self.errors(l_output)

    def save_MLP(self, f_output):
        '''Save the MLP into a file'''
        pickle.dump(self, open(f_output, "wb"))

    def open_MLP(self,f_input):
        '''Return the MLP saved in the file'''
        self = pickle.load(open(f_input,"rb"))
