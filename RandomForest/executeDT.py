import sys
import pickle
from TrainingSet import *
import matplotlib.pyplot as pyplot
import numpy


class Points(object):
    __slots__ = ("bolts","nuts","rings","scraps")
    def __init__(self):
        """initializes the point set to an empty collection of points"""
        self.bolts = list()
        self.nuts = list()
        self.rings = list()
        self.scraps = list()
    def plot(self,x,y,classGuess):
        """sets a point position to show on the plot when show() is called"""
        classGuess = int(classGuess)
        if classGuess == 1:
            self.bolts.append((x,y))
            return
        elif classGuess == 2:
            self.nuts.append((x,y))
            return
        elif classGuess == 3:
            self.rings.append((x,y))
            return
        elif classGuess == 4:
            self.scraps.append((x,y))
            return
        else:
            print ("class not valid: "+str(classGuess))
            return
    def show(self):
        '''Plot the data with the given class'''
        for p in self.bolts:
            pyplot.plot([p[0]], [p[1]],'ro')
        for p in self.nuts:
            pyplot.plot([p[0]], [p[1]],'bo')
        for p in self.rings:
            pyplot.plot([p[0]], [p[1]],'go')
        for p in self.scraps:
            pyplot.plot([p[0]], [p[1]],'yo')
        pyplot.axis([0, 1, 0, 1])
        pyplot.xlabel('x1')
        pyplot.ylabel('x2')
        pyplot.show()



def main():
    #command line usage validation
    if len(sys.argv) != 3:
        print ("Usage: python trainDT.py <ForestFile> <TestingCSVFilename>")
        return

    #create forest from file
    forestFilename = sys.argv[1]
    forest = pickle.load(open(forestFilename,"rb"))

    #constants
    ATTRIBUTE_TITLES = [1,2]
    PROFIT_MATRIX = [[0.2,-0.07,-0.07,-0.07],\
                     [-0.07,0.15,-0.07,-0.07],\
                     [-0.07,-0.07,0.05,-0.07],\
                     [-0.03,-0.03,-0.03,-0.03]]

    #these variables are the points for our decisions.
    #note that some of the points may be misclassified
    points = Points()
    confusionMatrix = [[0,0,0,0],\
                       [0,0,0,0],\
                       [0,0,0,0],\
                       [0,0,0,0]]

    #create testing sample set from file
    filename = sys.argv[2]
    reader = csv.reader(open(filename,'rb'),delimiter=',')
    examples = list()
    for row in reader:
        examples.append(stringsToTrainingExample(row,ATTRIBUTE_TITLES))
    exSet = ExampleSet(examples)

    errorCount = 0
    totalCount = 0
    profit = 0.0
    for example in exSet.examples:
        sample = dict()
        for att in ATTRIBUTE_TITLES:
            sample[att] = example.attributes[att]
        result = forest.classify(sample)#the guess provided by our forest
        totalCount += 1
        if result != example.classValue:#note: example.classValue=correct class
            errorCount += 1
        profit += PROFIT_MATRIX[int(result)-1][int(example.classValue)-1]
        points.plot(sample[ATTRIBUTE_TITLES[0]],\
                    sample[ATTRIBUTE_TITLES[1]],\
                    result)
        confusionMatrix[int(result)-1][int(example.classValue)-1] += 1
    recRate = 1.0 - float(errorCount)/float(totalCount)

    print "There are " + str(totalCount-errorCount) + \
          " correctly classified samples and " + \
          str(errorCount) + " misclassified samples."
    print "The recognition rate is " + str(recRate) + "."
    print "The profit is " + str(profit) + "."
    print ""
    print "The confusion matrix is:"
    for row in confusionMatrix:
        print row
    points.show()

if "__main__" == __name__:
    main()
    
