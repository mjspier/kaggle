# Helper Class to validata classifiers
# Author: Manuel Spierenburg


import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from xgboost import XGBClassifier

# ignore warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)



def run(X, y, n_folds=10):
    _runKNeighborsClassifier(X, y, n_folds)
    _runSVCClassifier(X, y, n_folds)
    _runRandomForestClassifier(X, y, n_folds)
    _runDecisionTreeClassifier(X, y, n_folds)
    _runExtraTreesClassifier(X, y, n_folds)
    _runAdaBoostClassifier(X, y, n_folds)
    _runGaussianNB(X, y, n_folds)
    _runLinearDiscriminantAnalysis(X, y, n_folds)
    _runQuadraticDiscriminantAnalysis(X, y, n_folds)
    #_runGradientBoostingClassifier(X, y, n_folds)
    _runXGBClassifier(X, y, n_folds)

    
def _runKNeighborsClassifier(X, y, n_folds):
    print('Validate KNeighborsClassifier')
    # find best parameters
    best_k = 0
    best_metric = 0
    best_algorithm = 0

    best_score = 0
    for k in [1,2,3,4,5,10,100,200,500,1000]:
        if k > len(X):
            continue
        for algo in ['ball_tree','kd_tree','brute']:
             for me in ['euclidean', 'minkowski', 'chebyshev', 
                        'manhattan','minkowski', 'wminkowski',
                        'seuclidean','mahalanobis']:
                 try:
                     model = KNeighborsClassifier(n_neighbors=k,
                                     algorithm=algo,
                                     metric=me)
                     scores = cross_val_score(model, X, y, cv=n_folds)
                 except:
                     continue
                 m = np.mean(scores)
                 #print('k:%d a:%s m:%s score:%f ' % (k,algo,me,m))
                 if m > best_score:
                     best_score = m
                     best_k = k
                     best_algorithm = algo
                     best_metric = me
    
    print('score:%f #config k:%d a:%s m:%s' % 
          (best_score, best_k, best_algorithm, best_metric))
    return best_score, best_k, best_algorithm, best_metric


def _runSVCClassifier(X, y, n_folds):
    print('Validate SVC')
    # find best parameters
    best_k = None
    best_score = 0
    # kernel
    for k in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
        model = SVC(kernel=k, max_iter=1000000)
        try:
            scores = cross_val_score(model, X, y, cv=n_folds)
        except:
            continue
        m = np.mean(scores)
        #print('k:%s score:%f' % (k,m))
        if m > best_score:
            best_score = m
            best_k = k
    
    print('score:%f #config k:%s' % 
          (best_score, best_k))
    return best_score, best_k


def _runRandomForestClassifier(X, y, n_folds):
    print('Validate RandomForestClassifier')
    # find best parameters
    best_n = 0
    best_maxf = 0

    best_score = 0
    # number of trees
    for n in [1,2,3,4,5,10,100,200,500,1000]:
        # number of features for split
        for maxf in [0.1, 0.2, 0.5, 'auto','sqrt','log2', None]:
            model = RandomForestClassifier(n_estimators=n,
                                     max_features=maxf)
            scores = cross_val_score(model, X, y, cv=n_folds)
                
            m = np.mean(scores)
            #print('n:%d maxf:%s score:%f ' % (n,maxf,m))
            if m > best_score:
                best_score = m
                best_n = n
                best_maxf = maxf
    
    print('score:%f #config n:%d maxf:%s' 
          % (best_score, best_n, best_maxf))
    return best_score, best_n, best_maxf


def _runDecisionTreeClassifier(X, y, n_folds):
    print('Validate DecisionTreeClassifier')
    # find best parameters
    best_c = 0
    best_s = 0
    best_maxf = 0
    best_score = 0
    
    # criterion
    for c in ['gini', 'entropy']:
        # splitter
        for s in ['best', 'random']:
            # number of features for split
            for maxf in [0.1, 0.2, 0.5, 'auto','sqrt','log2', None]:
                model = DecisionTreeClassifier(criterion=c,
                                     splitter=s,
                                     max_features=maxf)
                scores = cross_val_score(model, X, y, cv=n_folds)
                
                m = np.mean(scores)
                #print('c:%d s:%d maxf:%s score:%f ' % (c,s,maxf,m))
                if m > best_score:
                    best_score = m
                    best_c = c
                    best_maxf = maxf
                    best_s = s
    
    print('score:%f #config c:%s s:%s maxf:%s' 
          % (best_score, best_c, best_s, best_maxf))
    return best_score, best_c, best_s, best_maxf


def _runExtraTreesClassifier(X, y, n_folds):
    print('Validate ExtraTreesClassifier')
    # find best parameters
    best_c = 0
    best_n = 0
    best_maxf = 0

    best_score = 0
    
    # number of trees
    for n in [1,2,3,4,5,10,100,200,500,1000]:
        # criterion
        for c in ['gini', 'entropy']:
            # number of features for split
            for maxf in [0.1, 0.2, 0.5, 'auto','sqrt','log2', None]:
                model = ExtraTreesClassifier(n_estimators=n,
                                     criterion=c,
                                     max_features=maxf)
                scores = cross_val_score(model, X, y, cv=n_folds)
                
                m = np.mean(scores)
                #print('c:%d s:%d maxf:%s score:%f ' % (c,s,maxf,m))
                if m > best_score:
                    best_score = m
                    best_c = c
                    best_maxf = maxf
                    best_n = n
    
    print('score:%f #config n:%s c:%s maxf:%s' 
          % (best_score, best_n, best_c, best_maxf))
    return best_score, best_n, best_c, best_maxf

def _runAdaBoostClassifier(X, y, n_folds):
    print('Validate AdaBoostClassifier')
    # find best parameters
    best_n = 0
    best_score = 0
    
    # number of boosters
    for n in [10,20,50,100,200,500,1000,10000]:
        model = AdaBoostClassifier(n_estimators=n)
        scores = cross_val_score(model, X, y, cv=n_folds)
        m = np.mean(scores)
        if m > best_score:
            best_score = m
            best_n = n
    
    print('score:%f #config n:%s' 
          % (best_score, best_n))
    return best_score, best_n


def _runGaussianNB(X, y, n_folds):
    print('Validate GaussianNB')
    model = GaussianNB()
    scores = cross_val_score(model, X, y, cv=n_folds)
    m = np.mean(scores)    
    print('score:%f' % (m))
    return m


def _runLinearDiscriminantAnalysis(X, y, n_folds):
    print('Validate LinearDiscriminantAnalysis')
    model = LinearDiscriminantAnalysis()
    scores = cross_val_score(model, X, y, cv=n_folds)
    m = np.mean(scores)    
    print('score:%f' % (m))
    return m


def _runQuadraticDiscriminantAnalysis(X, y, n_folds):
    print('Validate QuadraticDiscriminantAnalysis')
    model = QuadraticDiscriminantAnalysis()
    scores = cross_val_score(model, X, y, cv=n_folds)
    m = np.mean(scores)    
    print('score:%f' % (m))
    return m


def _runGradientBoostingClassifier(X, y, n_folds):
    print('Validate GradientBoostingClassifier')
     # find best parameters
    best_n = 0
    best_d = 0
    best_score = 0
    # number of trees
    for n in [10,20,50,100,200,500,1000,10000]:
        # max depth
        for d in [1,2,3,5,10,20,50,100]:
            model = GradientBoostingClassifier(n_estimators =n)
            scores = cross_val_score(model, X, y, cv=n_folds)
            m = np.mean(scores) 
            if m > best_score:
                best_score = m
                best_n = n
                best_d = d
    print('score:%f #config n:%d d:%d' % (best_score, best_n, best_d))
    return best_score, best_n, best_d

def _runXGBClassifier(X, y, n_folds):
    print('Validate XGBClassifier')
     # find best parameters
    best_n = 0
    best_score = 0
    # number of trees
    for n in [10,20,50,100,200,500,1000,10000]:
        model = XGBClassifier(n_estimators =n)
        scores = cross_val_score(model, X, y, cv=n_folds)
        m = np.mean(scores) 
        if m > best_score:
            best_score = m
            best_n = n
    print('score:%f #config n:%d' % (best_score, best_n))
    return best_score, best_n
