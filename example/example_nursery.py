# Dataset https://archive.ics.uci.edu/ml/datasets/Nursery

import numpy as np
import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import mean_squared_error, make_scorer, roc_auc_score, log_loss
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from racog import RACOG

RS = 334

nurseryurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data'
attribute_list = ['parents', 'has_nurs', 'form', 'children',
                  'housing', 'finance', 'social', 'health', 'target']

nursery = pd.read_csv(nurseryurl, header=None, names=attribute_list)

LE = LabelEncoder()

X = nursery.drop('target', axis=1)
y = nursery['target']

ii = y[y == 'recommend'].index.values
X.drop(ii, inplace=True)
y.drop(ii, inplace=True)

for col in X:
    if X[col].dtype == 'object':
        X[col] = LE.fit_transform(X[col])
X = X.values

LE = LabelEncoder()
y = LE.fit_transform(y)

rf = RandomForestClassifier()
params = {'class_weight': 'balanced',
          'criterion': 'entropy',
          'max_depth': 15,
          'max_features': 0.9,
          'min_samples_leaf': 11,
          'min_samples_split': 2,
          'min_weight_fraction_leaf': 0,
          'n_estimators': 30}
rf.set_params(**params)

gscore = make_scorer(geometric_mean_score, average='multiclass')


def gmean(y_true, y_pred):
    return geometric_mean_score(y_true, y_pred, average='multiclass')


strf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)
count = 0
for train_index, test_index in strf.split(X, y):
    print(Counter(y[test_index]), Counter(y[train_index]))
    # swap train/test
    X_train, X_test, y_train, y_test = X[test_index], X[train_index], y[test_index], y[train_index]
    rf.set_params(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('#####################################################')
    print('Count', count)
    print('')
    print('Without oversampling | Gmean:', gmean(y_test, y_pred))
    rnd_over = RandomOverSampler(random_state=RS + count)

    X_rndo, y_rndo = rnd_over.fit_sample(X_train, y_train)
    print('')
    rf.fit(X_rndo, y_rndo)
    y_pred = rf.predict(X_test)
    print('Random oversampling | Gmean:', gmean(y_test, y_pred))

    smote = SMOTE(random_state=RS + count, kind='regular', k_neighbors=5, m=None,
                  m_neighbors=10, n_jobs=1)
    X_smote, y_smote = smote.fit_sample(X_train, y_train)

    rf.fit(X_smote, y_smote)
    y_pred = rf.predict(X_test)
    print('')
    print('SMOTE oversampling | Gmean:', gmean(y_test, y_pred))

    racog = RACOG(categorical_features='all',
                  warmup_offset=100, lag0=20, n_iter='auto',
                  threshold=10, eps=10E-5, verbose=0, n_jobs=1)

    X_racog, y_racog = racog.fit_sample(X_train, y_train)
    rf.fit(X_racog, y_racog)
    y_pred = rf.predict(X_test)

    print('RACOG oversampling | Gmean:', gmean(y_test, y_pred))
    print('')
    count = count + 1
