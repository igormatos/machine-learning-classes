import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import arff	
    
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

    df = pd.DataFrame(data=dataArff['data'], columns=attrs_t)

    # Normalizar
    scaler = StandardScaler()
    scaler.fit(df.drop('defects@NUMERIC',axis=1))
    scaled_features = scaler.transform(df.drop('defects@NUMERIC',axis=1))
    
    df_fitted = pd.DataFrame(scaled_features,columns=df.columns[:-1])
    
    x = scaled_features
    y = df['defects@NUMERIC']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(classification_report(y_test,pred))

    # Plot also the training points
    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, 'uniform'))
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

    plt.show()

main()