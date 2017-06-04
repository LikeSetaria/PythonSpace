# -*- coding: utf8 -*-
import numpy as np
import urllib.request
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# url with dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
# download the file
raw_data = urllib.request.urlopen(url).read()
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset)
# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)

model = LogisticRegression()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))