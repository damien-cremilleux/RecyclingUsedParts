"""
trainDT.py

Authors:
    Joseph Fuchs
    Damien Cremilleux
"""

from copy import deepcopy
from copy import copy
import sys
import csv
import math
import random

from TrainingSet import *
from EnsembleForest import *

import pickle
import matplotlib.pyplot as pyplot
import numpy

################################################################################
class DTLNode(object):
    """DTLNode interface"""
    def classify(self, sample):
        """
        sample: a structure that can access attribute values by: 
                sample[attName]
        """
        raise Exception("unimplemented method: DTLNode.classify")

################################################################################
class ClassNode(DTLNode):
    """
    A DTL leaf node that simply holds a class value
    """
    __slots__ = ("classValue",)
    def __getstate__(self):
        """gets the state of the node as a tuple"""
        return (self.classValue,)
    def __setstate__(self,stateTuple):
        self.classValue = stateTuple[0]

    def __init__(self,classVal):
        self.classValue = classVal
    def classify(self, sample):
        return self.classValue
    def __str__(self):
        return str(self.classValue)

################################################################################
class BinaryDTLNode(DTLNode):
    """
    An object for handling continious valued decision trees
    
    attributeName - the name of the attribute
    splitValue - the value to split the continuous variables into two 
                 partitions.
                 A > splitValue ; B <= splitValue
    greaterThan - the BinaryDTLNode or class value associated with 
    """
    __slots__ = ("attributeName","splitValue","greaterThan","lessThanOrEqual")
    def __getstate__(self):
        """gets the state of the tree as a tuple"""
        return (self.attributeName,self.splitValue,self.greaterThan,self.lessThanOrEqual)
    def __setstate__(self,stateTuple):
        """sets the state of the forest from a tuple"""
        self.attributeName = stateTuple[0]
        self.splitValue = stateTuple[1]
        self.greaterThan = stateTuple[2]
        self.lessThanOrEqual = stateTuple[3]

    def __init__(self,attributeName,split,gt,ltet):
        self.attributeName = attributeName
        self.splitValue = split
        self.greaterThan = gt
        self.lessThanOrEqual = ltet
    def classify(self, sample):
        if sample[self.attributeName] > self.splitValue:
            return self.greaterThan.classify(sample)
        else:
            return self.lessThanOrEqual.classify(sample)
    def __str__(self):
        s = str(self.attributeName)
        s += "\n{"
        s += "\n>" + str(self.splitValue) + ":" + str(self.greaterThan)
        s += "\n<=" + str(self.splitValue) + ":" + str(self.lessThanOrEqual)
        s += "\n}"
        return s

################################################################################
def decisionTreeLearning(exampleSet,attributes,default):
    """
    Returns: a Decision Tree that was trained from the given example list
    Parameters:
        exampleSet - an ExampleSet of TrainingExamples
        attributes - a list of attributes names.  These should be the keys in
                   an example that map to the value of a feature.
        default  - our default value; if classes are defaulted to this if they
                   aren't found a fit otherwise.
    """
    ###if examples is empty, then return default
    if exampleSet.size() == 0:
        #print "Base: no examples left, default=",default
        return ClassNode(default)
    
    ###else, if all examples have the same class, return that class
    if exampleSet.allHaveSameClass():
        #print "Base: examples have same class: ",exampleSet.examples[0].classValue
        return ClassNode(exampleSet.examples[0].classValue)
    
    ###else, if attributes is empty, then return mode(examples)
    mode = exampleSet.mode()
    if len(attributes) == 0:
        #print "Base: no attributes left, mode=",mode
        return ClassNode(mode)
    
    ###else: recurse
    best = exampleSet.chooseBestAttribute(attributes)
    bestAttribute = best[0]
    bestSplit = best[1]
    #print "recursing"
    #print "    Best Attr:  ", bestAttribute
    #print "    Best Split: ", bestSplit
    #print "    Mode:       ", mode
      ###find LTET tree
    subExamples = exampleSet.subExampleSetLTET(bestAttribute,bestSplit)
    #subAttributes = deepcopy(attributes)
    subAttributes = copy(attributes)
    subAttributes.remove(bestAttribute)
    ltetTree = decisionTreeLearning(subExamples,subAttributes,mode)
      ###find GT tree
    subExamples = exampleSet.subExampleSetGT(bestAttribute,bestSplit)
    #subAttributes = deepcopy(attributes)
    subAttributes = copy(attributes)
    subAttributes.remove(bestAttribute)
    gtTree = decisionTreeLearning(subExamples,subAttributes,mode)

    tree = BinaryDTLNode(bestAttribute,bestSplit,gtTree,ltetTree)
    return tree

################################################################################
def chooseRandomCombination(inputList,n):
    """chooses a random combination of n elements from a list"""
    lst = deepcopy(inputList)
    chosen = list()
    size = len(lst)
    x = 0
    while x < n:
        randInt = random.randint(0,size-1-x)
        chosen.append(lst[randInt])
        lst[randInt] = lst[size-1-x]
        x += 1
    return chosen

################################################################################
def dtlRandom(exampleSet,attributes,default,mTry):
    """
    Returns: a Decision Tree that was trained from the given example list
    Parameters:
        exampleSet - an ExampleSet of TrainingExamples
        attributes - a list of attributes names.  These should be the keys in
                   an example that map to the value of a feature.
        default  - our default value; if classes are defaulted to this if they
                   aren't found a fit otherwise.
    """
    ###if examples is empty, then return default
    if exampleSet.size() == 0:
        #print "Base: no examples left, default=",default
        return ClassNode(default)
    
    ###else, if all examples have the same class, return that class
    if exampleSet.allHaveSameClass():
        #print "Base: examples have same class: ",exampleSet.examples[0].classValue
        return ClassNode(exampleSet.examples[0].classValue)
    
    ###else, if attributes is empty, then return mode(examples)
    mode = exampleSet.mode()
    if len(attributes) == 0:
        #print "Base: no attributes left, mode=",mode
        return ClassNode(mode)
    
    ###else: recurse
    #####Change from normal DTL####
    ##***we must also randomly select from attributes***##
    m = min(mTry,len(attributes))
    randAtts = chooseRandomCombination(attributes,m)

    best = exampleSet.chooseBestAttribute(randAtts)#changed from original
    bestAttribute = best[0]
    bestSplit = best[1]
    #print "recursing"
    #print "    Best Attr:  ", bestAttribute
    #print "    Best Split: ", bestSplit
    #print "    Mode:       ", mode
      ###find LTET tree
    subExamples = exampleSet.subExampleSetLTET(bestAttribute,bestSplit)
    #subAttributes = deepcopy(attributes)
    subAttributes = copy(attributes)
    subAttributes.remove(bestAttribute)
    ltetTree = dtlRandom(subExamples,subAttributes,mode,mTry)
      ###find GT tree
    subExamples = exampleSet.subExampleSetGT(bestAttribute,bestSplit)
    #subAttributes = deepcopy(attributes)
    subAttributes = copy(attributes)
    subAttributes.remove(bestAttribute)
    gtTree = dtlRandom(subExamples,subAttributes,mode,mTry)

    tree = BinaryDTLNode(bestAttribute,bestSplit,gtTree,ltetTree)
    return tree
###############################################################################

def plotError(forest,exampleSet,ssePlots,sizePlots):
    """
    Augments datapoints to lists of plotting coordinates.
    This does not plot or display data.

    forest - the forest whose data is being mapped to a plot
    examples - the test examples used to test the forest
    ssePlots - list of Sum of Squared Error to be plotted (Y-axis)
        Error is defined as the fraction of incorrect votes for a sample.
    sizePlots - list of X coordinates corresponding to SSE's
    """
    sizePlots.append(forest.getSize())
    sse = 0.0 #sum of squared error
    for examp in exampleSet.examples:
        votes = forest.tallyVotes(examp.attributes)
        totalVoteCount = 0
        for cls in votes:
            totalVoteCount += votes[cls]
        if not examp.classValue in votes:
            correctVoteCount = 0
        else:
            correctVoteCount = votes[examp.classValue]
        fractionCorrect = (float(correctVoteCount)/float(totalVoteCount))
        difference = 1.0 - fractionCorrect #aka, fraction of wrong votes
        sse += difference*difference
    ssePlots.append(sse)
    

###############################################################################
##Python Version 2.7
##>python trainDT.py <TrainingCSVFilename>

def main():
    #command line usage validation
    if len(sys.argv) != 2:
        print ("Usage: python trainDT.py <TrainingCSVFilename>")
        return

    #constants
    MAX_TREES = 200         #size of forest
    ATTRIBUTE_TITLES = [1,2] #these won't affect functionality
    SF = (lambda s: 2*int(2*math.sqrt(s)))#2sqrt(size) sample size from max
    OUT = [1,10,100,MAX_TREES] #which training sets to output

    #holds our SSE for each 
    ssePlots = list()
    sizePlots = list()

    #read training file to ExampleSet
    filename = sys.argv[1]
    reader = csv.reader(open(filename,'rb'),delimiter=',')
    examples = list()
    for row in reader:
        examples.append(stringsToTrainingExample(row,ATTRIBUTE_TITLES))
    exSet = ExampleSet(examples)

    sampleSize = SF(len(exSet.examples))
    forest = RandomForest(0,0,exSet,ATTRIBUTE_TITLES)

    #populate forest, and ouput trees when necessary
    for curTree in range(1,MAX_TREES+1):
        #add one more tree to the forest
        forest.addTrees(1,sampleSize,exSet,ATTRIBUTE_TITLES)
        #plot error to pyplot
        plotError(forest,exSet,ssePlots,sizePlots)
        if curTree in OUT:
            #output this current tree to file
            outfile = "dt"+str(curTree)+".p"
            pickle.dump(forest,open(outfile,"wb"))

    #plot the datapoints that have been accumulated by plotError().
    pyplot.plot(sizePlots,ssePlots)
    pyplot.ylabel("Sum of squared errors")
    pyplot.xlabel("Number of trees")
    pyplot.show()

if __name__ == "__main__":
    main()
