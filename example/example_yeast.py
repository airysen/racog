# Dataset from https://archive.ics.uci.edu/ml/datasets/Car+Evaluation


import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVR, SVC
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import mean_squared_error, make_scorer, log_loss
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler, MinMaxScaler, normalize


from sklearn.neighbors import KNeighborsClassifier
from noisePGA import NoiseGAEnsemble

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from racog import RACOG


def target_convert(a):
    if a == 'EXC':
        return 1
    else:
        return 0


RS = 335

yeast_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'

yeastdf = pd.read_csv(yeast_url, delim_whitespace=True)

attribute_list = ['seq', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'target']

yeastdf.columns = attribute_list
yeastdf.drop(['seq', 'erl', 'pox'], axis=1, inplace=True)
X = yeastdf.drop('target', axis=1)

for col in X:
    if X[col].dtype == 'object':
        LE = LabelEncoder()
        X[col] = LE.fit_transform(X[col])


y = yeastdf['target'].apply(target_convert)

X = X.values
y = y.values

rf = RandomForestClassifier()
params = {'class_weight': 'balanced',
          'criterion': 'entropy',
          'max_depth': 15,
          'max_features': 0.7,
          'min_samples_leaf': 11,
          'min_samples_split': 10,
          'min_weight_fraction_leaf': 0,
          'n_estimators': 30,
          'random_state': RS}


gscore = make_scorer(geometric_mean_score, average='binary')


def gmean(y_true, y_pred):
    return geometric_mean_score(y_true, y_pred, average='binary')


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

    smote = SMOTE(random_state=RS + count, kind='regular', k_neighbors=2, m=None,
                  m_neighbors=10, n_jobs=1)
    X_smote, y_smote = smote.fit_sample(X_train, y_train)

    rf.fit(X_smote, y_smote)
    y_pred = rf.predict(X_test)
    print('')
    print('SMOTE oversampling | Gmean:', gmean(y_test, y_pred))

    racog = RACOG(discretization='caim', categorical_features='auto',
                  warmup_offset=100, lag0=20, n_iter='auto',
                  continous_distribution='normal',
                  alpha=0.6, L=0.7, threshold=10, eps=10E-5, verbose=0, n_jobs=1)

    X_racog, y_racog = racog.fit_sample(X_train, y_train)
    rf.fit(X_racog, y_racog)
    y_pred = rf.predict(X_test)

    print('RACOG oversampling | Gmean:', gmean(y_test, y_pred))
    print('')
    count = count + 1
