# pylint: disable=unused-import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# prepocessing tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# classifiers
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# evaluation tools
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, balanced_accuracy_score

dataset = pd.read_csv("features/features_and_label.csv")

X = dataset.iloc[:,1:51].values
y = dataset.iloc[:,71:81].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    # TODO real split

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train_int = np.argmax(y_train, axis=1)
y_test_int = np.argmax(y_test, axis=1)

weights = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)
class_weights = dict(enumerate(weights))

tuned_parameters = {
    'bootstrap': [False],
    'max_depth': [50, 100, 200, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 4, 6],
    'min_samples_split': [5, 10, 20],
    'n_estimators': [50, 500, 1000, 2500],
    'class_weight': [class_weights]
}

estimator = RandomForestClassifier()

estimator_random = GridSearchCV(estimator=estimator, param_grid=tuned_parameters, 
            n_jobs=-1, cv=2, verbose=2, scoring='balanced_accuracy')

estimator_random.fit(X_train, y_train_int)
print(estimator_random.best_params_)
res = estimator_random.cv_results_
print(res)

np.save('parameters.npy', res)