"""
GMM Accuracy with 200 features 0.33 with k fold
GMM Accuracy with 400 features 0.24 with k fold

Bayes GMM with 200 features 0.23
Bayes GMM with 400 features 0.24

"""

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split
import pandas
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold


file_path_200f = "/home/k_mathin/PycharmProjects/DataMiningClass/datasets/classification/object_classification/yin_airpl_m_sun/target_6_4_101_91_MICC_F220_bow_200.csv"
#
data = pandas.read_csv("/home/k_mathin/PycharmProjects/DataMiningClass/datasets/classification/heart.csv",
                       header=None,
                       skiprows=1)
# data = pandas.read_csv(file_path_200f, header=None, skiprows=1)

X = data.iloc[:, 0:len(data.keys()) - 1]  # slicing: all rows and 1 to 4 cols
# store response vector in "y"
y = data.iloc[:, len(data.keys()) - 1]
k = len(np.unique(y))
X = np.asarray(X)
y = np.asarray(y)
skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(X, y)
accuracy_list = []

for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    mixture_model = GaussianMixture(n_components=k)
    mixture_model.fit(X_train, y_train)
    predictions = mixture_model.predict(X_test)



    accuracy_list.append(sklearn.metrics.accuracy_score(y_test, predictions))
print(accuracy_list)
print("Final acc: ", np.asarray(accuracy_list).mean())