import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
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

def random_prototype(train):
	n_records = len(train)
	n_features = len(train[0])
	prototype = [train[randrange(n_records)][i] for i in range(n_features)]
	return prototype

def train_prototypes(train, n_prototypes, learn_rate, epochs):
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

def main():
    n_neighbors = 1

    # Arff to dataframe
    dataArff = arff.load(open('cm1.arff', 'r'))
    attrs = dataArff['attributes']
    
    attrs_t = []
    for attr in attrs:
        if isinstance(attr[1], list):
            attrs_t.append("%s@{%s}" % (attr[0], ','.join(attr[1])))
        else:
            attrs_t.append("%s@%s" % (attr[0], attr[1]))

    ## Learning Vector Quantization
    lrate = 0.3
    n_epochs = 80
    n_prototypes = 20
    prototypes = train_prototypes(dataArff['data'], n_prototypes, lrate, n_epochs)

    # Transformando os dados em DataFrame
    df = pd.DataFrame(data=prototypes, columns=attrs_t)

    backup = pd.DataFrame(data=dataArff['data'], columns=attrs_t)
    backup_x = backup.drop('defects@NUMERIC',axis=1)
    backup_y = backup['defects@NUMERIC']
    train_x, test_x, train_y, test_y = train_test_split(backup_x, backup_y, test_size=0.30)

    # Normalizar as features
    scaler = StandardScaler()
    scaler.fit(df.drop('defects@NUMERIC',axis=1))
    scaled_features = scaler.transform(df.drop('defects@NUMERIC',axis=1))
    
    ## Dataset e divisao entre treino e teste
    x = scaled_features
    y = df['defects@NUMERIC']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

    # Instaciamos o classificador KNN
    knn = neighbors.KNeighborsClassifier(n_neighbors)
    
    # Checamos a relacao entre as features e o valor de y
    knn.fit(x_train, y_train)

    ## Prever com um novo conjunto, utilizando nosso knn
    print(classification_report(test_y, knn.predict(test_x)))

    # Plotar os pontos de treinamento
    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, 'uniform'))
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    # plt.scatter(np.asarray(test_x)[:, 0], np.asarray[:, 1], c=test_y, cmap=cmap_bold, edgecolor='k', s=5)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    
    plt.show()

main()