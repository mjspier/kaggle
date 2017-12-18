# Script for Kaggel Digit Recognizer Competition
# Author: Manuel Spierenburg

import numpy as np
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
# train: shape n x 785, first column are labels
# test:  shape n x 784, 784 pixel = 28x28 images
train = np.genfromtxt(train_file, delimiter=',', skip_header=1, dtype=int)
test = np.genfromtxt(test_file, delimiter=',', skip_header=1, dtype=int)

labels = train[:,0]
data = train[:,1:]

#####################
# get averages and show
#####################
for i in range(10):
    digits = data[labels==i,:]
    average = np.mean(digits, 0)
    x = np.reshape(average, (28, 28))
    plt.subplot(4, 3, i+1)
    plt.imshow(x)

plt.show()

#####################
# model selection
#####################
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
    scores = cross_val_score(clf, data, labels, cv=n_folds)
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
        scores = cross_val_score(clf, data, labels, cv=n_folds)
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

