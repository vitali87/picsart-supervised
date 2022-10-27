from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from skopt import BayesSearchCV

X,y = make_classification()

# X = pd.DataFrame(np_[0])
# y = pd.DataFrame(np_[1])

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)

sss.get_n_splits(X, y)

for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

LR = LogisticRegression(random_state=0).fit(X_train, y_train)
SVM = SVC(random_state=0, probability=True).fit(X_train, y_train)
NB = GaussianNB().fit(X_train, y_train)
KNN = KNeighborsClassifier().fit(X_train, y_train)

lr_param = {
    "penalty": ['l1','l2', 'elasticnet'],
    "C": [0.5, 1],
    "l1_ratio": [0.5]
}

knn_param ={
    'weights': ['uniform', 'distance'],
    'n_neighbors': [3, 5, 7]

}

svm_param = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}



lr_pred = LR.predict(X_test)
svm_pred = SVM.predict(X_test)
NB.predict(X_test)
KNN.predict(X_test)

lr_pred_proba = LR.predict_proba(X_test)
svm_pred_proba = SVM.predict_proba(X_test)
NB.predict_proba(X_test)
KNN.predict_proba(X_test)

scores = [LR.score(X_test,y_test),
          SVM.score(X_test,y_test),
          NB.score(X_test,y_test),
          KNN.score(X_test,y_test)]


opt_svc = BayesSearchCV(
    SVC(random_state=0, probability=True),
    svm_param,
    n_iter=32,
    random_state=0
)

_ = opt_svc.fit(X_train, y_train)
print(opt_svc.score(X_test, y_test))

opt_lr = BayesSearchCV(
    LogisticRegression(random_state=0, solver="saga"),
    lr_param,
    n_iter=32,
    random_state=0
)

_ = opt_lr.fit(X_train, y_train)
print(opt_lr.score(X_test, y_test))