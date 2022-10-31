from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=10**6, n_features=30, n_classes=100, n_informative=10, random_state=301
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=301
)

rf = RandomForestClassifier(random_state=301, n_jobs=-1, max_depth=12, verbose=1)
rf_ovo = RandomForestClassifier(random_state=301, max_depth=12, verbose=1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
rf_f1 = f1_score(y_test, pred_rf, average="macro")
rf_acc = accuracy_score(y_test, pred_rf)

clf_RF_ovr = OneVsRestClassifier(rf_ovo, verbose=1, n_jobs=-1).fit(X_train, y_train)
pred_rf_ovo = clf_RF_ovr.predict(X_test)
rf_f1_ovo = f1_score(y_test, pred_rf_ovo, average="macro")
rf_acc_ovo = accuracy_score(y_test, pred_rf_ovo)

# lr = LogisticRegression(random_state=301, solver="saga")
dt = DecisionTreeClassifier(random_state=301, max_depth=12)
# clf_LR_ovr = OneVsRestClassifier(lr, verbose=2, n_jobs=-1).fit(X_train, y_train)
clf_DT_ovr = OneVsRestClassifier(dt, verbose=2, n_jobs=-1).fit(X_train, y_train)
# clf_LR_ovo = OneVsOneClassifier(lr, n_jobs=-1).fit(X_train, y_train)
# clf_LR_oc = OutputCodeClassifier(estimator=LogisticRegression(random_state=301),
#                            random_state=301).fit(X, y)

# pred_lr_ovr = clf_LR_ovr.predict(X_test)
# pred_lr_ovo = clf_LR_ovo.predict(X_test)
pred_dt_ovr = clf_DT_ovr.predict(X_test)
# pred_lr_oc = clf_LR_oc.predict(X_test)

# clf_LR_native = LogisticRegression(
#     random_state=301, n_jobs=-1, multi_class="ovr", solver="saga", verbose=2
# ).fit(X_train, y_train)
# pred_lr_native = clf_LR_native.predict(X_test)

# lr_f1_ovr = f1_score(y_test, pred_lr_ovr, average="macro")
# lr_acc_ovr = accuracy_score(y_test, pred_lr_ovr)
# print(f"f1: {lr_f1_ovr}")
# print(f"acc: {lr_acc_ovr}")

# lr_f1_ovo = f1_score(y_test, pred_lr_ovo, average="macro")
# lr_acc_ovo = accuracy_score(y_test, pred_lr_ovo)
# print(f"f1: {lr_f1_ovo}")
# print(f"acc: {lr_acc_ovo}")

dt_f1_ovo = f1_score(y_test, pred_dt_ovr, average="macro")
dt_acc_ovo = accuracy_score(y_test, pred_dt_ovr)
print(f"f1: {dt_f1_ovo}")
print(f"acc: {dt_acc_ovo}")

# lr_f1_oc = f1_score(y_test, pred_lr_oc, average="macro")
# lr_acc_oc = accuracy_score(y_test, pred_lr_oc)
# print(f"f1: {lr_f1_oc}")
# print(f"acc: {lr_acc_oc}")

# lr_ovr
# f1: 0.09786793037894423
# f1: 0.07845550166000442
# acc: 0.090235

# lr_ovo
# f1: 0.10501832072905139
# acc: 0.113555

# lr_oc
# f1: 0.02794181630147891
# acc: 0.047685

# dt_ovo (no depth constraint)
# f1: 0.1296781251431877
# acc: 0.114405
#
# dt_ovo (max depth 12)
# f1: 0.20307517135473918
# acc: 0.204445

# rf (max depth 12)
# rf_f1: 0.2977184925094594
# rf_acc: 0.29984

# rf_ovr
# f1: 0.2983215870382313
# acc: 0.301585
