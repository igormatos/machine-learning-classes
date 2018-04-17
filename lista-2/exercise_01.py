import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import arff
import mglearn
from random import randrange
from random import seed


def best_matching_vector(prototypes, test_row):
	distances = list()
	for prototype in prototypes:
		dist = np.linalg.norm(np.array(prototype) - np.array(test_row))
		distances.append((prototype, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

def best_matching_vectorr(prototypes, test_row):
	distances = list()
	for prototype in prototypes:
		dist = np.linalg.norm(np.array(prototype) - np.array(test_row))
		distances.append((prototype, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0], distances[1][0]


def random_prototype(train):
	n_records = len(train)
	n_features = len(train[0])
	prototype = [train[randrange(n_records)][i] for i in range(n_features)]
	return prototype


def lqv1(train, n_prototypes, learn_rate, epochs):
	prototypes = [random_prototype(train) for i in range(n_prototypes)]
	for epoch in range(epochs):
		rate = learn_rate * (1.0-(epoch/float(epochs)))
		sum_error = 0.0
		for row in train:
			bmu = best_matching_vector(prototypes, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				sum_error += error**2
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
	return prototypes

def lqv21(train, n_prototypes, learn_rate, epochs):
	prototypes = [random_prototype(train) for i in range(n_prototypes)]
	for epoch in range(epochs):
		rate = learn_rate * (1.0-(epoch/float(epochs)))
		sum_error = 0.0
		for row in train:
			bmuA, bmuB = best_matching_vectorr(prototypes, row)
			if in_window(row, bmuA, bmuB):
				if bmuA[-1] != bmuB[-1]:
					for i in range(len(row)-1):
						error = row[i] - bmuA[i]
						sum_error += error**2

						if bmuA[-1] == row[-1]:  # row = x / bmuA = Ei
							bmuA[i] += rate * error
							bmuB[i] -= rate * error
						else:
							bmuA[i] -= rate * error
							bmuB[i] += rate * error
	return prototypes

def in_window(row, prot_a, prot_b):
	w = 0.3
	s = (1-w)/(1+w)
	di = np.linalg.norm(np.array(prot_a) - np.array(row))
	dj = np.linalg.norm(np.array(prot_b) - np.array(row))
	didj = di/dj
	djdi = dj/di
	
	return np.min([didj, djdi]) > s



def main():
    n_neighbors = 3

    # Arff to dataframe
    dataArff = arff.load(open('cm1.arff', 'r'))
    attrs = dataArff['attributes']

    attrs_t = []
    for attr in attrs:
        if isinstance(attr[1], list):
            attrs_t.append("%s@{%s}" % (attr[0], ','.join(attr[1])))
        else:
            attrs_t.append("%s@%s" % (attr[0], attr[1]))

    # Learning Vector Quantization
    useLqv = 1 #deixando false, nao e usado o LQV

    # Transformando os dados em DataFrame
    df = pd.DataFrame(data=dataArff['data'], columns=attrs_t)

    if useLqv:
        lrate = 0.3
        n_epochs = 80
        n_prototypes = 20
		
		# trocar lqv21 por lqv1
        prototypes = lqv21(dataArff['data'], n_prototypes, lrate, n_epochs)

        df = pd.DataFrame(data=prototypes, columns=attrs_t)

    # Backup do dataset
    backup = pd.DataFrame(data=dataArff['data'], columns=attrs_t)
    backup_x = backup.drop('defects@NUMERIC', axis=1)
    backup_y = backup['defects@NUMERIC']
    train_x, test_x, train_y, test_y = train_test_split(
        backup_x, backup_y, test_size=0.30)

    # Normalizar as features
    scaler = StandardScaler()
    scaler.fit(df.drop('defects@NUMERIC', axis=1))
    scaled_features = scaler.transform(df.drop('defects@NUMERIC', axis=1))

    # Dataset e divisao entre treino e teste
    x = scaled_features
    y = df['defects@NUMERIC']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

    # Instaciamos o classificador KNN
    knn = neighbors.KNeighborsClassifier(n_neighbors)

    # Checamos a relacao entre as features e o valor de y
    knn.fit(x_train, y_train)

    # Prever com um novo conjunto, utilizando nosso knn
    pred = knn.predict(test_x)
    print(classification_report(test_y, pred))
    
    # Plotar os pontos de treinamento
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold=ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, 'uniform'))
    # plt.scatter(np.asarray(x_train)[:, 0], np.asarray(x_train)[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=10)
    # plt.scatter(np.asarray(backup_x)[:, 0], np.asarray(backup_x)[:, 1], c=backup_y, cmap=cmap_light, edgecolor='k', s=10)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=30)

    plt.show()

main()

# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]
# learn_rate = 0.3
# n_epochs = 10
# n_codebooks = 20
# codebooks = lqv21(dataset, n_codebooks, learn_rate, n_epochs)
# print(len(codebooks))