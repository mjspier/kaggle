# Script for Kaggel Titanic Competitions
# Author: Manuel Spierenburg

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
import matplotlib.pyplot as plt
import seaborn as sns

# config
# number of folds in cross validation
n_folds = 10

train_file = './data/train.csv'
test_file = './data/test.csv'

#####################
# prepare data
#####################
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

#####################
# visualiziation 
#####################


survived = train.Survived

# plot relationships
plt.figure()
sns.countplot(x="Sex", hue="Survived",  data=train)
plt.figure()
sns.boxplot(x='Survived', y='Fare', data=train)
plt.figure()
sns.boxplot(x='Survived', y='Age', data=train)
plt.figure()
sns.countplot(x="SibSp", hue="Survived",  data=train)
plt.figure()
sns.countplot(x="Parch", hue="Survived",  data=train)
plt.figure()
sns.countplot(x="Pclass", hue="Survived",  data=train)
plt.figure()
sns.countplot(x="Embarked", hue="Survived",  data=train)



#####################
# data cleansing 
#####################

# select features
fts = ['Pclass','Sex','Age','SibSp','Parch','Fare']
X_train = train[fts]
X_test = test[fts]

X_train['Sex'] = X_train['Sex'] == 'male'
X_train = X_train.fillna(X_train.Age.mean())

X_test['Sex'] = X_test['Sex'] == 'male'
X_test = X_test.fillna(X_train.Age.mean())


#####################
# model selection
#####################
models = [
    ['KNeighborsClassifier',KNeighborsClassifier()],
    ['SVC',SVC()],
    ['RandomForestClassifier',RandomForestClassifier()],
    ['DecisionTreeClassifier',DecisionTreeClassifier()],
    ['ExtraTreesClassifier',ExtraTreesClassifier()],
    ['AdaBoostClassifier',AdaBoostClassifier()],
    ['GaussianNB',GaussianNB()],
    ['LinearDiscriminantAnalysis',LinearDiscriminantAnalysis()],
    ['QuadraticDiscriminantAnalysis',QuadraticDiscriminantAnalysis()]]

# iterate over models
results = {}
for name, model in models:
    print('Model: ',name)
    scores = cross_val_score(model, X_train, train.Survived, cv=n_folds)
    #print scores
    m = np.mean(scores)
    print('mean score: ', m)
    results[name] = m

#######################
# random forrest was best, make first submission
#######################
m = RandomForestClassifier()
m.fit(X_train, train.Survived)
s = m.score(X_train, train.Survived)
print("score: ", s)
y_test = m.predict(X_test)

sub = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": y_test})
sub.to_csv("submission_randomforest.csv", index=False)
