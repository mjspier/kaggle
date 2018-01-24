# Helper Class to validata linear models
# Author: Manuel Spierenburg

from util import run_tests
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, 
                                 ElasticNet, BayesianRidge, Lasso,
                                 LassoCV, LassoLars, LassoLarsCV)

# ignore warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_simple(X, y, n_folds=10):
    models = [LinearRegression, Ridge, RidgeCV, ElasticNet, BayesianRidge,
              Lasso, LassoCV ,LassoLars ,LassoLarsCV]
    for m in models:
        run_tests(m, {}, X, y, n_folds)


def run(X, y, n_folds=10):
    _runLinearRegression(X, y, n_folds)
    _runRidge(X, y, n_folds)
    _runRidgeCV(X, y, n_folds)
    _runElasticNet(X, y, n_folds)
    _runBayesianRidge(X, y, n_folds)
    _runLasso(X, y, n_folds)
    _runLassoCV(X, y, n_folds)
    _runLassoLars(X, y, n_folds)
    _runLassoLarsCV(X, y, n_folds)


def _runLinearRegression(X, y, n_folds):
    params = {'normalize' : [False, True]}
    run_tests(LinearRegression, params, X, y, n_folds)
    

def _runRidge(X, y, n_folds):
    params = {'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 
                          'sparse_cg', 'sag', 'saga']}
    run_tests(Ridge, params, X, y, n_folds)
    
    
def _runRidgeCV(X, y, n_folds):
    params = {'normalize' : [False, True]}
    run_tests(RidgeCV, params, X, y, n_folds)
    

def _runElasticNet(X, y, n_folds):
    params = {'normalize' : [False, True]}
    run_tests(ElasticNet, params, X, y, n_folds)
    
    
def _runBayesianRidge(X, y, n_folds):
    params = {'normalize' : [False, True],
              'n_iter' : [300, 500, 1000]}
    run_tests(BayesianRidge, params, X, y, n_folds)
    
    
def _runLasso(X, y, n_folds):
    params = {'normalize' : [False, True]}
    run_tests(Lasso, params, X, y, n_folds)
    
    
def _runLassoCV(X, y, n_folds):
    params = {'normalize' : [False, True]}
    run_tests(LassoCV, params, X, y, n_folds)
    
    
def _runLassoLars(X, y, n_folds):
    params = {'normalize' : [False, True]}
    run_tests(LassoLars, params, X, y, n_folds)
    
    
def _runLassoLarsCV(X, y, n_folds):
    params = {'normalize' : [False, True]}
    run_tests(LassoLarsCV, params, X, y, n_folds)