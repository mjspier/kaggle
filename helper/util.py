# Utilities for helper classes
# Author: Manuel Spierenburg

import sys
import numpy as np
import itertools
from sklearn.model_selection import cross_val_score


def run_tests(model, param_dict, X, y, n_folds):
    param_keys = list(param_dict.keys())
    param_lists = list(param_dict.values())
    best_params = dict()
    best_score = -sys.maxsize
    # get all combinations
    for params in itertools.product(*param_lists):
        args = dict()
        for i in range(len(params)):
            args[param_keys[i]] = params[i]
        try:
            m = model(**args)
            scores = cross_val_score(m, X, y, cv=n_folds)
        except:
            e = sys.exc_info()[0]
            #print('ERROR: %s PARAMS: %s' % (e, args))
            continue          
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = args
    print('score:%f %s params:%s' % (best_score, model.__name__, best_params))

