# Script for Kaggel Titanic Competitions
# Author: Manuel Spierenburg


# autoreload for ipyhton
#%load_ext autoreload
#%autoreload 2

import AutoClassifiers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

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
total = pd.concat([train, test])

# use median age for missing ages
total.Age.fillna(total.Age.median(), inplace=True)
train = total[:len(train)]

#####################
# visualiziation 
#####################


survived = train.Survived

# plot relationships
plt.figure()
sns.countplot(x="Sex", hue="Survived",  data=train)
# female are more likely to survive

plt.figure()
plt.hist([train[train.Survived == 1].Age, train[train.Survived == 0].Age], 
         stacked=True, bins=30, color=['g','r'], label=['Survived','Died'])
# children <= 10 more likely to survive


plt.figure()
plt.hist([train[train.Survived == 1].Fare, train[train.Survived == 0].Fare], 
         stacked=True, bins=30, color=['g','r'], label=['Survived','Died'])
# higher fare >~70 more likely to survive


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

# extract title from name attribute
titles = [n.split(',')[1].split('.')[0].strip() for n in total.Name]
total['Title'] = titles
total.Title.value_counts()

# convert title to numeric value
title_map =  {v: k for k,v in enumerate(np.unique(titles))}
total.Title = total.Title.map(title_map).astype(int)

# change sex to boolean 
total['Sex'] = total['Sex'] == 'male'

# create new family features
total['FamilySize'] = total.SibSp + total.Parch
total['IsAlone'] = total['FamilySize'] == 0

# fill missing Fare value with 0
total.Fare.fillna(0, inplace=True)
total.Embarked.fillna('S', inplace=True)
emb_map =  {v: k for k,v in enumerate(np.unique(total.Embarked))}
total.Embarked = total.Embarked.map(emb_map).astype(int)


# predict age for missing rows with linear regression
#x = total[~total.Age.isnull()]
#fts = ['Pclass','Sex','SibSp','Parch','Fare', 'Title']
#m = RidgeCV()
#m.fit(x[fts], x.Age)
#print('lr age score:', m.score(x[fts], x.Age))
#
#x2 = total[total.Age.isnull()]
#y = m.predict(x2)

total.Age.fillna(total.Age.mean(), inplace=True)

# select features
fts = ['Pclass','Sex','Age','SibSp','Parch','Fare', 'Title', 'FamilySize', 
       'IsAlone', 'Embarked']
X_total = total[fts]

# split in train and test
X_train = X_total[:len(train)]
X_test = X_total[len(train):]

#####################
# model selection
#####################
AutoClassifiers.run(X_train, train.Survived, n_folds)



#######################
# random forest was best, make first submission
#######################
m = XGBClassifier(n_estimators = 200)
m.fit(X_train, train.Survived)
s = m.score(X_train, train.Survived)
print("score: ", s)
y_test = m.predict(X_test)

sub = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": y_test})
sub.to_csv("submission_xgboost.csv", index=False)
