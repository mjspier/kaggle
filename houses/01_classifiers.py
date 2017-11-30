# Script for Kaggel House Prices Competitions
# Author: Manuel Spierenburg

import sys, os, fnmatch
import operator
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import tree
from sklearn.externals.six import StringIO
import scipy.spatial.distance
import matplotlib.pyplot as plt

# config
# number of folds in cross validation
n_folds = 10

train_file = './data/train.csv'
test_file = './data/test.csv'

#####################
# prepare data
#####################
print('1. load data')
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

labels = train.SalePrice
data = train[train.columns.difference(['SalePrice'])]

#####################
# model selection
#####################
print('2. run different classifiers')
classifiers = [
    ['KNeighborsClassifier',KNeighborsClassifier()],
    ['SVC',SVC()],
    ['RandomForestClassifier',RandomForestClassifier()],
    ['DecisionTreeClassifier',DecisionTreeClassifier()],
    ['ExtraTreesClassifier',ExtraTreesClassifier()],
    ['AdaBoostClassifier',AdaBoostClassifier()],
    ['GaussianNB',GaussianNB()],
    ['LinearDiscriminantAnalysis',LinearDiscriminantAnalysis()],
    ['QuadraticDiscriminantAnalysis',QuadraticDiscriminantAnalysis()]]

# iterate over classifiers
results = {}
for name, classifier in classifiers:
    print('Classifier: ',name)
    scores = cross_val_score(classifier, data, labels, cv=n_folds)
    #print scores
    print('mean score: ',np.mean(scores))

# KNeighbours is best classifier
# try different metrics
metrics = ['euclidean','cosine','braycurtis','canberra','cityblock']
for metric in metrics:
    print('Metric: ',metric)
    clf = KNeighborsClassifier(metric=metric, algorithm='brute')
    scores = cross_validation.cross_val_score(clf, data, labels, cv=n_folds)
    print('mean score: ',np.mean(scores))

# metric: cosine was best
# try different number of neighbors
print('3. find good params for knn')
ks = [1,2,3,4,5,10,15,20,30,50,100]
ws = ['uniform','distance']

for w in ws:
    for k in ks:
        print('Weights: ',w)
        print('Number of neighbors: ',k)
        clf = KNeighborsClassifier(metric='cosine', algorithm='brute', n_neighbors=k, weights=w)
        scores = cross_validation.cross_val_score(clf, data, labels, cv=n_folds)
        print('mean score: ',np.mean(scores))


#######################
# knn submission
#######################
print('4. knn submission file')
c = KNeighborsClassifier(metric='cosine', algorithm='brute', n_neighbors=4)
c.fit(data,labels)
s = c.score(data, labels)
print("score: ", s)
pred = c.predict(test)
np.savetxt('submission_knn.csv', np.c_[range(1,len(test)+1),pred],
        delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


#######################
# random forest submission
#######################
print('5. random forest submission file')
c = RandomForestClassifier(n_estimators=1000, bootstrap=False)
c.fit(data,labels)
s = c.score(data, labels)
print("score: ", s)
pred = c.predict(test)
np.savetxt('submission_random.csv', np.c_[range(1,len(test)+1),pred],
        delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

