
from TrainingSet import *
from trainDT import *



class RandomForest(object):
    __slots__ = ("trees",)

    def __getstate__(self):
        """gets the state of the forest as a tuple"""
        return (self.trees,)
    def __setstate__(self,stateTuple):
        """sets the state of the forest from a tuple"""
        self.trees = stateTuple[0]
    def __init__(self,numTrees,sampleSize,totalTrainingSet,attTitles):
        self.trees = list()
        for t in range(numTrees):
            subSet = totalTrainingSet.randomSubset(sampleSize)
            mTry = int( math.sqrt(len(attTitles)) )#should evaluate to 1 for our examples
            newTree = dtlRandom(subSet,attTitles,0,mTry)
            self.trees.append(newTree)
    def getSize(self):
        return len(self.trees)
    def classify(self,sample):
        """
        sample is a map:    sample[attTitle] -> attValue
        """
        ###all trees in the forest vote
        votes = self.tallyVotes(sample)
        ###our final classification is the plurality vote
        highestCount = 0
        highestClass = None
        for cls in votes:
	    val = votes[cls]
            if val > highestCount:
                highestCount = val
                highestClass = cls
	#print votes
        return highestClass
    def tallyVotes(self,sample):
        """creates a map:  Classification -> #Votes ; float -> int"""
        votes = dict()
        for t in self.trees:
            result = t.classify(sample)
            if not result in votes:
                votes[result] = 1
            else:
                votes[result] += 1
        return votes
    def addTrees(self,numTrees,sampleSize,totalTrainingSet,attTitles):
        """adds numTrees trees to this forest"""
        for t in range(numTrees):
            subSet = totalTrainingSet.randomSubset(sampleSize)
            mTry = int( math.sqrt(len(attTitles)) )#should evaluate to 1 for our examples
            newTree = dtlRandom(subSet,attTitles,0,mTry)
            self.trees.append(newTree)



if __name__ == "__main__":
    filename = sys.argv[1]
    reader = csv.reader(open(filename,'rb'),delimiter=',')
    examples = list()
    attributeTitles = [1,2]
    for row in reader:
        examples.append(stringsToTrainingExample(row,attributeTitles))
    exSet = ExampleSet(examples)

    numTrees = int(sys.argv[2])

    sampleSize = int(2*math.sqrt(len(exSet.examples)))
    forest = RandomForest(numTrees,sampleSize,exSet,attributeTitles)

    errorCount = 0
    totalCount = 0
    for example in exSet.examples:
        
        sample = dict()
        for att in attributeTitles:
            sample[att] = example.attributes[att]
        result = forest.classify(sample)
        totalCount += 1
        if result != example.classValue:
            errorCount += 1
        print "Expected: ", example, " \t Got: ", result
    print "Errors: ",errorCount
    print "Total:  ",totalCount
    print float(errorCount)/float(totalCount)*100.0, "%"
