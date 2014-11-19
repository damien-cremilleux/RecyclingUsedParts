#!/usr/bin/env python

import Neural_Network as NN
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

"""
trainMLP.py

Train a multi layer perceptron

Authors:
    Joseph Fuchs <jjf2614@rit.edu>
    Damien Cremilleux <dxc9849@rit.edu>
"""


def plot_errors(errors):
    '''Plot the sum of squared errors'''
    plt.plot(errors)
    plt.xlabel('Number of weight updates')
    plt.ylabel('Sum of squared errors')
    plt.show()

if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit("Error: wrong number of args")
    
    data = []
    nb_epoch = 10000
    errors = []
    
    # Read the data
    csvFile = sys.argv[1]
    with open(csvFile, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for x,y,c in spamreader:
            data.append([x,y,c]);
        
    # Create and train the MLP
    mlp = NN.MLP(2,5,4)
    for ii in range(0, nb_epoch):
        sum_errors =0 
        for lp in data:
            # Generate inputs
            l_input = []
            for jj in range(0,len(lp)-1):
                l_input.append(float(lp[jj]))

            # Generate output
            l_output = []
            if lp[len(lp)-1] == '1':
                l_output = [1, 0, 0, 0]
            if lp[len(lp)-1] == '2':
                l_output = [0, 1, 0, 0]
            if lp[len(lp)-1] == '3':
                l_output = [0, 0, 1, 0]
            if lp[len(lp)-1] == '4':
                l_output = [0, 0, 0, 1]
         
            e  = mlp.train_MLP(l_input, l_output)
            sum_errors = sum_errors + e

        errors.append(sum_errors)
            
        # Save the MLP at given epoch
        mlp.inc_epoch()
        e = mlp.get_epoch()
        if e == 10:
            mlp.save_MLP("MLP_10")
        if e == 100:
            mlp.save_MLP("MLP_100")
        if e == 1000:
            mlp.save_MLP("MLP_100")
        if e == 1000:
            mlp.save_MLP("MLP_1000")
        if e == 10000:
            mlp.save_MLP("MLP_10000")

    # Plot the errors
    plot_errors(errors) 
        
        
