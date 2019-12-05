from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import datasets
import pandas
import numpy as np

iris = datasets.load_breast_cancer()


X = iris.data
y = iris.target

### heart disease accuracy .52
"""
data = pandas.read_csv("/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/classification/heart.csv", header=None,
                       skiprows=1)


X = data.iloc[:, 0:13]  # slicing: all rows and 1 to 4 cols
# store response vector in "y"
y = data.iloc[:, 13]
"""

# pulsar star 87%
data = pandas.read_csv("/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/classification/pulsar_stars.csv", header=None,
                       skiprows=1)


X = data.iloc[:, 0:7]  # slicing: all rows and 1 to 4 cols
# store response vector in "y"
y = data.iloc[:, 8]




k = 2
X = np.asarray(X)
y = np.asarray(y)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
mixture_model = GaussianMixture(n_components=2)
mixture_model.fit(X_train, y_train)
predictions = mixture_model.predict(X_test)
sklearn.metrics.accuracy_score(y_test, predictions)




