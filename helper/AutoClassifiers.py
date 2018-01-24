# Helper Class to validata classifiers
# Author: Manuel Spierenburg


from util import run_tests
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
    _runGradientBoostingClassifier(X, y, n_folds)
    _runXGBClassifier(X, y, n_folds)
    
    
def _runKNeighborsClassifier(X, y, n_folds):
    params = {'n_neighbors' : [1,2,3,4,5,10,100,200,500,1000],
              'algorithm' : ['ball_tree','kd_tree','brute'],
              'metric' : ['euclidean', 'minkowski', 'chebyshev', 
                        'manhattan','minkowski', 'wminkowski',
                        'seuclidean','mahalanobis']}
    run_tests(KNeighborsClassifier, params, X, y, n_folds)


def _runSVCClassifier(X, y, n_folds):
    params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
              'max_iter':[1000000]}
    run_tests(SVC, params, X, y, n_folds)


def _runRandomForestClassifier(X, y, n_folds):
    params = {'n_estimators': [1,2,3,4,5,10,100,200,500,1000],
              'max_features': [0.1, 0.2, 0.5, 'auto','sqrt','log2', None]}
    run_tests(RandomForestClassifier, params, X, y, n_folds)


def _runDecisionTreeClassifier(X, y, n_folds):
    params = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_features': [0.1, 0.2, 0.5, 'auto','sqrt','log2', None]}
    run_tests(DecisionTreeClassifier, params, X, y, n_folds)


def _runExtraTreesClassifier(X, y, n_folds):
    params = {'n_estimators': [1,2,3,4,5,10,100,200,500,1000],
              'criterion': ['gini', 'entropy'],
              'max_features': [0.1, 0.2, 0.5, 'auto','sqrt','log2', None]}
    run_tests(ExtraTreesClassifier, params, X, y, n_folds)
    

def _runAdaBoostClassifier(X, y, n_folds):
    params = {'n_estimators': [10,20,50,100,200,500,1000]}
    run_tests(AdaBoostClassifier, params, X, y, n_folds)


def _runGaussianNB(X, y, n_folds):
    run_tests(GaussianNB, {}, X, y, n_folds)


def _runLinearDiscriminantAnalysis(X, y, n_folds):
    run_tests(LinearDiscriminantAnalysis, {}, X, y, n_folds)


def _runQuadraticDiscriminantAnalysis(X, y, n_folds):
    run_tests(QuadraticDiscriminantAnalysis, {}, X, y, n_folds)


def _runGradientBoostingClassifier(X, y, n_folds):
    params = {'n_estimators': [10,20,50,100,200,500,1000],
              'max_depth' : [3,5,10,20,50,100]}
    run_tests(GradientBoostingClassifier, params, X, y, n_folds)


def _runXGBClassifier(X, y, n_folds):
    params = {'n_estimators': [10,20,50,100,200,500,1000]}
    run_tests(XGBClassifier, params, X, y, n_folds)
