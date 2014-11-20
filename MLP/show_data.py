#!/usr/bin/env python

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

"""
show_data.py

Show the data given in parameter

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

    if len(sys.argv) != 2:
        sys.exit("Error: wrong number of args")

    # Read the data file
    csvFile = sys.argv[1]
    with open(csvFile, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for x,y,c in spamreader:
            data.append([float(x),float(y),float(c)]);

    # Classify the data
    l_bolt = []
    l_nut = []
    l_ring = []
    l_scrap = []
 
    for d in data:
        # Generate output
        if d[len(d)-1] == 1:
            l_bolt.append(d)
        elif d[len(d)-1] == 2:
            l_nut.append(d)
        elif d[len(d)-1] == 3:
            l_ring.append(d)
        elif d[len(d)-1] == 4:
            l_scrap.append(d)

    # Plot the data
    plot_data(l_bolt, l_nut, l_ring, l_scrap)
    

                   
               
