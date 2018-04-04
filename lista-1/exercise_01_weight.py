import random
import math
import operator
import arff	
import matplotlib.pyplot as plot
import numpy as numpy
import csv
import matplotlib.patches as mpatches


def loadData(filename, split, trainingSet=[], testSet=[]):
	data = arff.load(open(filename, 'r'))
	dataset = list(data['data'])
	for x in range(len(dataset)-1):
		for y in range(4):
			dataset[x][y] = float(dataset[x][y])
		if random.random() < split:
			trainingSet.append(dataset[x])
		else:
			testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = {}
	for x in range(k):
		neighbors[distances[x][1]] = distances[x][0]
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for distance, neighbor in neighbors.items():
		response = neighbor[-1]
		if response in classVotes:
			classVotes[response] += 1 * (1/float(distance))
		else:
			classVotes[response] = 1/float(distance) if distance != 0 else 1
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
	
	# k_values = [1,2,3,5,7,9,11,13,15]
	k_values = range(1,40,2)

	for k in k_values:
		trainingSet=[]
		testSet=[]
		loadData('defect.arff', split, trainingSet, testSet)
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