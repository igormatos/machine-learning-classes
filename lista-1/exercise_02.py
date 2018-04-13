import random
import math
import operator
import arff	
import matplotlib.pyplot as plot
import numpy as numpy
import csv

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = dataset[x][y]
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])

# def loadData(filename, split, trainingSet=[], testSet=[]):
# 	data = arff.load(open(filename, 'r'))
# 	dataset = list(data['data'])
# 	for x in range(len(dataset)-1):
# 		for y in range(4):
# 			dataset[x][y] = float(dataset[x][y])
# 		if random.random() < split:
# 			trainingSet.append(dataset[x])
# 		else:
# 			testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def valueDifferenceMetric(testSet, trainingSet, classes):
    # Return distance between two values x,y
    #  of an attribute a
    distance = 0
    for n in range(1,4):
        distance += vdmI(testSet, trainingSet, classes, n)

    return distance

def vdmI(testSet, trainingSet, classes, a):
    _x = 2
    _y = 1
    distance = 0
    
    for c in classes:
        print "c" + c
        print "a" + repr(a)
        print "x[a]" + repr(testSet)
        ax = [x for x in trainingSet if x[1] == _x]
        print "ax: " + repr(ax)
        nax = len(ax)
        print "nax: " + nax

        axc = [x for x in ax if x[0] == c]
        naxc = len(axc)
        
        paxc = float(naxc) / float(nax)

        ay = [x for x in trainingSet if x[a] == _y]
        nay = len(ay)
        
        ayc = [x for x in ay if x[0] == c]
        nayc = len(ayc)

        payc = float(nayc) / float(nay)

        distance += paxc-payc

    return distance


def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	classes = list(set([i[0] for i in trainingSet]))

	for x in range(len(trainingSet)):
		dist = valueDifferenceMetric(testInstance, trainingSet[x], classes)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	split = 0.67
	error_rate = []
	
	k_values = range(1,5,4)

	for k in k_values:
		trainingSet=[]
		testSet=[]
		loadDataset('balance.data', split, trainingSet, testSet)
		predictions=[]
		for x in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[x], k)
			result = getResponse(neighbors)
			predictions.append(result)
		accuracy = getAccuracy(testSet, predictions)
		error_rate.append(accuracy)
	
	plot.figure(figsize=(10,6))
	plot.plot(k_values,error_rate,color='blue', linestyle='dashed', marker='o',
			markerfacecolor='red', markersize=10)
	plot.title('Success Rate vs. K Value')
	plot.xlabel('K')
	plot.ylabel('Success Rate')
	print(numpy.mean(error_rate))
	plot.show()

main()