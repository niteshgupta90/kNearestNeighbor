import csv
import random
import numpy
from random import randint
import operator
import math
import matplotlib.pyplot as plt

def normalize(dataset=[]):
    dataset = numpy.array(dataset)
    mu =[]
    sd = []
    mu = numpy.mean(dataset, 0)
    sd = numpy.std(dataset, axis=0)
    
    for x in range(len(dataset)):
        for y in range(1,14):
            dataset[x][y] = float(dataset[x][y]-mu[y])/sd[y]
    return dataset

def main():
    # prepare data
    with open('wine.data.txt', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        
        for x in range(len(dataset)):
            for y in range(14):
                dataset[x][y] = float(dataset[x][y])
    dataset = normalize(dataset)
    totalWrong = dataSet = 0
    A = []
    B = []
    for k in range(1,10):
        for i in range(1,10):
            trainingSet=[]
            testSet=[]
            trainingSet, testSet = loadDataset(dataset, trainingSet, testSet)
            #print 'Train set: ' + repr(len(trainingSet))
            #print 'Test set: ' + repr(len(testSet))

            # generate predictions
            predictions=[]
            for x in range(len(testSet)):
                neighbors = getNeighbors(trainingSet, testSet[x], k)
                result = getResponse(neighbors)
                predictions.append(result)
                #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][0]))
            wrong = getError(testSet, predictions)
            totalWrong += wrong
            dataSet += len(testSet)
        prob = (totalWrong/float(dataSet)) * 100.0
        print('Error Probability for k = '+ repr(k) +' is ' + repr(prob) + '%')
        A.append(k)
        B.append(prob)
    plt.plot(A,B)
    plt.show()

def getError(testSet, predictions):
    wrong = 0
    for x in range(len(testSet)):
        if testSet[x][0] != predictions[x]:
            wrong += 1
    #print 'Wrongly Classified:' + repr(wrong)
    return wrong

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(1,length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def loadDataset(dataset=[], trainingSet=[] , testSet=[]):
        class1=[]
        class2=[]
        class3=[]
        for x in range(len(dataset)):
            if dataset[x][0] == 1:
                class1.append(dataset[x])
            elif dataset[x][0] == 2:
                class2.append(dataset[x])
            elif dataset[x][0] == 3:
                class3.append(dataset[x])

        l1 = len(class1)
        l2 = len(class2)
        l3 = len(class3)
        class1 = numpy.array(class1)
        class2 = numpy.array(class2)
        class3 = numpy.array(class3)
        
        training_idx = numpy.random.randint(l1, size=l1-5)
        test_idx = numpy.random.randint(l1, size=5)
        training, test = class1[training_idx,:], class1[test_idx,:]
        testSet = testSet + test.tolist()
        trainingSet = trainingSet + training.tolist()

        training_idx = numpy.random.randint(l2, size=l2-5)
        test_idx = numpy.random.randint(l2, size=5)
        training, test = class2[training_idx,:], class2[test_idx,:]
        testSet = testSet + test.tolist()
        trainingSet = trainingSet + training.tolist()

        training_idx = numpy.random.randint(l3, size=l3-5)
        test_idx = numpy.random.randint(l3, size=5)
        training, test = class3[training_idx,:], class3[test_idx,:]
        testSet = testSet + test.tolist()
        trainingSet = trainingSet + training.tolist()
        return trainingSet, testSet

main()
