#!/usr/bin/env python

import Neural_Network as NN
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
executeMLP.py

Execute a multi layer perceptron on a data file

Authors:
    Joseph Fuchs <jjf2614@rit.edu>
    Damien Cremilleux <dxc9849@rit.edu>
"""


def plot_data(bolt, nut, ring, scrap):
    '''Plot the data with the given class'''
    for p in bolt:
        plt.plot([p[0]], [p[1]],'ro')
    for p in nut:
        plt.plot([p[0]], [p[1]],'bo')
    for p in ring:
        plt.plot([p[0]], [p[1]],'go')
    for p in scrap:
        plt.plot([p[0]], [p[1]],'yo')
  
    plt.axis([0, 1, 0, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ == "__main__":

    data = []

    if len(sys.argv) != 3:
        sys.exit("Error: wrong number of args")

    # Read the trained network
    mlp = pickle.load(open(sys.argv[1], "rb"))

    # Read the data file
    csvFile = sys.argv[2]
    with open(csvFile, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for x,y,c in spamreader:
            data.append([float(x),float(y),float(c)]);

    # Run the data on the neural network
    nb_missclassified = 0
    nb_correct = 0

    l_bolt = []
    l_nut = []
    l_ring = []
    l_scrap = []

    confusion_matrix = []
    for ii in range(0,4):
        confusion_matrix.append([0,0,0,0])
 
    for d in data:
        
         # Generate inputs
        l_input = []
        for jj in range(0,len(d)-1):
            l_input.append(d[jj])

        # Generate output
        l_output = []
        if d[len(d)-1] == 1:
            l_output = [1, 0, 0, 0]
        if d[len(d)-1] == 2:
            l_output = [0, 1, 0, 0]
        if d[len(d)-1] == 3:
            l_output = [0, 0, 1, 0]
        if d[len(d)-1] == 4:
            l_output = [0, 0, 0, 1]

        result =  mlp.propagate_inputs(l_input)
        
        # Select the correct ouput and produce the output
        r = result.index(max(result))
        s = l_output.index(max(l_output))

        if r == s:
            nb_correct = nb_correct +1
        else:
            nb_missclassified = nb_missclassified +1

        if r == 0:
            l_bolt.append(d)
        if r == 1:
            l_nut.append(d)
        if r == 2:
            l_ring.append(d)
        if r == 3:
            l_scrap.append(d)

        confusion_matrix[r][s] = confusion_matrix[r][s] + 1

    # Calculate the profit
    profit = 0
    profit_matrix = [[0.2,-0.07,-0.07,-0.07],[-0.07,0.15,-0.07,-0.07],[-0.07,-0.07,0.05,-0.07],[-0.03,-0.03,-0.03,-0.03]]
    for ii in range(0,4):
        for jj in range(0,4):
            profit = profit + confusion_matrix[ii][jj] * profit_matrix[ii][jj]
    
    # Print the result
    recognition_rate = float(nb_correct) / (float(nb_correct) + float(nb_missclassified))
    print "There are " + str(nb_correct) + " correctly classified samples and " + str(nb_missclassified) + " missclassified samples."
    print "The recognition rate is " + str(recognition_rate) + "."
    print "The profit is " + str(profit) + "."
    print ""
    print "The confusion matrix is:"
    for row in confusion_matrix:
        print row

    # Plot classification region
    plot_data(l_bolt, l_nut, l_ring, l_scrap)
    

                   
               
