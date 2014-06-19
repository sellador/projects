# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:33:24 2014

@author: Seve
"""

from sklearn import datasets, metrics, cross_validation
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from matplotlib import pyplot as plt

load_data = 0
if load_data:
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = .3)

clf_list = {
    QDA(),
    LDA(),
    SVC(),
    KNN()
    }

for clf in clf_list:       
    clf.fit(X,y)

    y_pred = clf.predict(X_test)

    print "Method: {}".format(clf)
    print metrics.classification_report(y_test,y_pred)

