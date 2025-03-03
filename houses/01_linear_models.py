# Script for Kaggel House Prices Competitions
# Author: Manuel Spierenburg


import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import seaborn as sns
import AutoLinear

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

test_ids = test.Id
test.drop(['Id'], axis=1, inplace=True)
prices = train.SalePrice
data = train[train.columns.difference(['SalePrice','Id'])]

#####################
# visualiziation 
#####################
data.columns
sns.distplot(prices)
prices.describe()
print("Skewness: %f" % prices.skew())
print("Kurtosis: %f" % prices.kurt())

# plot relationsships with numeric features
numf = data.select_dtypes(include=[np.number]).columns
for f in numf:
    d = pd.concat([prices, data[f]], axis=1)  
    d.plot.scatter(x=f, y='SalePrice')

# plot relationships with categorical features
catf = data.select_dtypes(include=['object']).columns
for f in catf:
    d = pd.concat([prices, data[f]], axis=1)  
    plt.figure()
    sns.boxplot(x=f, y='SalePrice', data=d)
plt.show()

#####################
# data cleansing 
#####################

# convert categorical data to columns with 0 or 1
X_all = pd.concat([data, test])
X_all = pd.get_dummies(X_all)

# list columns with NaN
X_all.columns[X_all.isnull().any()]

# remove columns with NaN
X_all = X_all[X_all.columns.difference(X_all.columns[X_all.isnull().any()])]

# split into train and test again
X_train = X_all[:len(data)]
X_test = X_all[len(data):]


#####################
# model selection
#####################
AutoLinear.run(X_train, prices, n_folds)

#######################
# ridge cv was best, let's make first submission
#######################
m = RidgeCV()
m.fit(X_train, prices)
s = m.score(X_train, prices)
print("score: ", s)
y_test = m.predict(X_test)

sub = pd.DataFrame({"Id": test_ids, "SalePrice": y_test})
sub.to_csv("submission_ridgecv.csv", index=False)
