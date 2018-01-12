# Helper Class to validata linear models
# Author: Manuel Spierenburg

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, 
                                 ElasticNet, BayesianRidge, Lasso,
                                 LassoCV, LassoLars, LassoLarsCV)





def run(X, y, n_folds=10):
    models = [
    ['LinearRegression',LinearRegression()],
    ['Ridge',Ridge()],
    ['RidgeCV',RidgeCV()],
    ['ElasticNet',ElasticNet()],
    ['BayesianRidge',BayesianRidge()],
    ['Lasso',Lasso()],
    ['LassoCV',LassoCV()],
    ['LassoLars',LassoLars()],
    ['LassoLarsCV',LassoLarsCV()]]

    # iterate over models
    results = {}
    for name, model in models:
        scores = cross_val_score(model, X, y, cv=n_folds)
        #print scores
        m = np.mean(scores)
        print('score: %f model: %s' % (m, name))
        results[name] = m

