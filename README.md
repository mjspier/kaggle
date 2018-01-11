Kaggle scripts
==============

This is a playground for data analytics and machine learning tools which I use for Kaggle competitions. 


Helper
------

The helper module can be used in different projects to run experiments on preprocessed data. 
Install the module with pip to the environment which is used for the project.

```
cd helper
~/.virtalenvs/data/bin/pip install -e .
```

In IPyhton use the helper as follow

```
# autoreload module
%load_ext autoreload
%autoreload 2

import AutoClassifiers
AutoClassifiers.run(X_train, train.Survived, n_folds)
```
