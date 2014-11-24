"""
TrainingSet.py

Authors:
    Joseph Fuchs
    Damien Cremilleux

Description:
    
"""

import random
from copy import deepcopy
import sys
import csv
import math

################################################################################
class TrainingExample(object):
    """
    attributes - a map that maps attribute name to its value
    classValue - the classification that this example falll under
    """
    __slots__ = ("attributes","classValue")
    def __init__(self,attributeMap,classVal):
        self.attributes = attributeMap
        self.classValue = classVal
    ##def __getitem__(self,index):
    ##    return self.attributes[index]
    def __str__(self):
        s = ""
        for key in self.attributes:
            s += str(self.attributes[key]) + " "
        s += ": " + str(self.classValue)
        return s

################################################################################
class ExampleSet(object):
    """
    Container for a training set
    """
    __slots__ = ("examples",)
    def __init__(self,examples):
        """examples- a list of training examples"""
        self.examples = examples
    ##def __getitem__(self,index):
    ##    return examples[index]
    def entropy(self):
        """Calculates the entropy of this training set"""
        examples = self.examples
        ###first find probability distribution
        probTable = dict()
        count = 0
        for examp in examples:#first count all examples
            count += 1
            if not examp.classValue in probTable:
                probTable[examp.classValue] = 1
            else:
                probTable[examp.classValue] += 1
        for classification in probTable:#then divide example counts
            probTable[classification] = float(probTable[classification])/float(count)
        summation = 0.0
        for cls in probTable:
            p = probTable[cls]
            summation += p * (-math.log(p,2))
        return summation
    def subExampleSetLTET(self,attName,splitVal):
        """gets the subset of training samples less than or equal to split"""
        lst = list()
        for examp in self.examples:
            if examp.attributes[attName] <= splitVal:
                #lst.append(deepcopy(examp))
                lst.append(examp)
        newSet = ExampleSet(lst)
        return newSet
    def subExampleSetGT(self,attName,splitVal):
        """gets the subset of training samples greater than the split"""
        lst = list()
        for examp in self.examples:
            if examp.attributes[attName] > splitVal:
                #lst.append(deepcopy(examp))
                lst.append(examp)
        newSet = ExampleSet(lst)
        return newSet
    def size(self):
        """finds the number of samples in this training set"""
        return len(self.examples)
    def informationGain(self,att,split):
        """
        Finds the information gain given an attribute and a split threshold
        to partition continuous values of that attribute (aka. feature)
        """
        size = self.size()
        gt = self.subExampleSetGT(att,split)
        ltet = self.subExampleSetLTET(att,split)
        x = self.entropy()
        y = gt.entropy()*gt.size()/size
        z = ltet.entropy()*ltet.size()/size
        return x-y-z
    def findBestSplitPoint(self,att):
        """
        The best split point is the split point with highest information gain
        Note: finding this is very computation heavy.
        """
        #examples = deepcopy(self.examples)
        examples = self.examples
        examples.sort(key=(lambda examp: examp.attributes[att]))
        bestSplit = 0.0
        bestInfoGain = 0.0
        for i in range(1,len(examples)):
            if examples[i-1].attributes[att] != examples[i].attributes[att]:
                split = (examples[i-1].attributes[att] + examples[i].attributes[att])/2.0
                gain = self.informationGain(att,split)
                if bestInfoGain < gain:
                    bestInfoGain = gain
                    bestSplit = split
        return bestSplit
    def allHaveSameClass(self):
        """Returns true if all examples have the same classification"""
        if self.size() < 2:
            return True
        for i in range(1,self.size()):
            if self.examples[i-1].classValue != self.examples[i].classValue:
                return False
        return True
    def mode(self):
        """Finds the mode classification of this set"""
        count = dict()
        for examp in self.examples:
            if not examp.classValue in count:
                count[examp.classValue] = 1
            else:
                count[examp.classValue] += 1
        mostClass = None
        mostCount = 0
        for cls in count:
            if count[cls] > mostCount:
                mostCount = count[cls]
                mostClass = cls
        return mostClass
    def chooseBestAttribute(self,attributes):
        """
        Chooses the best attribute from the given subset of attributes.
        The best attribute is the one with highest information gain
        """
        bestIG = 0.0
        bestAttr = None
        bestSplit = None
        for attr in attributes:
            split = self.findBestSplitPoint(attr)
            newIG = self.informationGain(attr,split)
            if bestIG < newIG:
                bestIG = newIG
                bestAttr = attr
                bestSplit = split
        return (attr,split)
    def randomSubset(self,amt):
        """
        Creates a new training subset (possible repeated values.
        The new training set is constructed by picking samples and replacing
        them until the size of the new set is equal to amt.
        """
        newExamples = list()
        for samp in range(0,amt):
            randInt = random.randint(0,self.size()-1)
            #newExamp = deepcopy(self.examples[randInt])
            newExamp = self.examples[randInt]
            newExamples.append(newExamp)
        return ExampleSet(newExamples)



################################################################################
def stringsToFloats(row):
    """produces a list of floats from a list of strings"""
    newRow = list()
    for s in row:
        newRow.append(float(s))
    return newRow
################################################################################
def stringsToTrainingExample(row,attNames):
    floats = stringsToFloats(row)
    atts = dict()
    cls = floats[len(floats)-1]
    for i in range(len(floats)-1):
        atts[attNames[i]] = floats[i]
    return TrainingExample(atts,cls)
################################################################################
if __name__ == "__main__":
    from trainDT import *
    filename = sys.argv[1]
    reader = csv.reader(open(filename,'rb'),delimiter=',')
    examples = list()
    attributeTitles = [1,2]
    for row in reader:
        examples.append(stringsToTrainingExample(row,attributeTitles))
    exSet = ExampleSet(examples)

    tree = decisionTreeLearning(exSet,attributeTitles,0)

    errorCount = 0
    totalCount = 0
    for example in exSet.examples:
        
        sample = dict()
        for att in attributeTitles:
            sample[att] = example.attributes[att]
        result = tree.classify(sample)
        totalCount += 1
        if result != example.classValue:
            errorCount += 1
        #print "Expected: ", example, " \t Got: ", result
    print "Errors: ",errorCount
    print "Total:  ",totalCount
    print float(errorCount)/float(totalCount)*100.0, "%"
    #print tree
    
