from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")
Train = data.drop(["Name"], axis=1)

Train[['Cabin']] = Train[['Cabin']].fillna('missing')
Train[['Embarked']] = Train[['Embarked']].fillna('missing')

# This is done for LightGBM
for c in Train.columns:
    col_type = Train[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        Train[c] = Train[c].astype('category')

X_train = Train.drop(["Survived"], axis=1)
y_train = Train["Survived"]

X_test = pd.read_csv("test.csv")

#This is done for CatBoost
X_test[['Cabin']] = X_test[['Cabin']].fillna('missing')
X_test[['Embarked']] = X_test[['Embarked']].fillna('missing')
X_test.drop(["Name"], axis=1, inplace=True)

# This is done for LightGBM
for c in X_test.columns:
    col_type = X_test[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        X_test[c] = X_test[c].astype('category')


# LightGBM cv
LGBM_clf = LGBMClassifier(random_state=0)
LGBM_cv = cross_val_score(LGBM_clf, X_train, y_train, cv=5)

# CatBoost cv
CatB_clf = CatBoostClassifier(random_state=0,
                              cat_features=[2, 6, 8, 9])
CatB_cv = cross_val_score(CatB_clf, X_train, y_train, cv=5)

# Pre-processing for Decision Tree and XGBoost
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train.iloc[:, [2, 6, 8, 9]]).toarray()
pd_X_train_enc = pd.DataFrame(X_train_enc)
pd_X_train_enc.columns = enc.get_feature_names_out(X_train.columns[[2, 6, 8, 9]])

X_train_proc = X_train.drop(X_train.columns[[2, 6, 8, 9]],axis=1)
X_train_proc = pd.concat([X_train_proc,pd_X_train_enc], axis=1)

# XGB cv
XGB_clf = XGBClassifier(random_state=0)
XGB_cv = cross_val_score(XGB_clf, X_train_proc, y_train, cv=5)

# DT cv
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
X_train_proc[["Age"]] = imp_median.fit_transform(X_train_proc[["Age"]])

DT_clf = DecisionTreeClassifier(random_state=0)
DT_cv = cross_val_score(DT_clf, X_train_proc, y_train, cv=5)

FINAL = [np.mean(x) for x in [LGBM_cv, CatB_cv, XGB_cv, DT_cv]]

# We choose CatBoost based on un-tuned models
CatB_clf.fit(X_train, y_train)
CatB_clf.predict(X_test)


# Now GridSearchCV

# Catboost
parameters = {'depth': [5, 10, 15],
              'learning_rate': [0.001, 0.01, 0.1],
              'iterations': [10, 50, 100]
              }
Grid_CBC = GridSearchCV(estimator=CatB_clf,
                        param_grid=parameters,
                        cv=5,
                        n_jobs=-1)

Grid_CBC.fit(X_train, y_train)

Grid_CBC.cv_results_
Grid_CBC.best_score_
Grid_CBC.best_params_

Grid_CBC.predict(X_test)

# XGBoost
parameters = {'max_depth': [5, 10, 15],
              'learning_rate': [0.001, 0.01, 0.1],
              'n_estimators': [300, 500, 1000]
              }
Grid_XGB = GridSearchCV(estimator=XGB_clf,
                        param_grid=parameters,
                        cv=5,
                        n_jobs=-1)

Grid_XGB.fit(X_train_proc, y_train)

Grid_XGB.cv_results_
Grid_XGB.best_score_
Grid_XGB.best_params_

Grid_XGB.predict(X_test)

# LGBM
parameters = {'max_depth': [15, 20, 25],
              'learning_rate': [0.005, 0.01, 0.02],
              'n_estimators': [100, 200, 300]
              }
Grid_LGBM = GridSearchCV(estimator=LGBM_clf,
                        param_grid=parameters,
                        cv=5,
                        n_jobs=-1)

Grid_LGBM.fit(X_train_proc, y_train)

Grid_LGBM.cv_results_
Grid_LGBM.best_score_
Grid_LGBM.best_params_

# This needs preprcoessing
Grid_LGBM.predict(X_test)