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

X = dataset.iloc[:,1:71].values
y = dataset.iloc[:,71:81].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    # TODO real split

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train_int = np.argmax(y_train, axis=1)
y_test_int = np.argmax(y_test, axis=1)

weights = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)
class_weights = dict(enumerate(weights))

tuned_parameters = {'kernel': ['rbf', 'poly'], 'gamma': [0.1, 1e-2, 1e-3, 'auto'],
                    'C':[1, 5, 10, 50, 100], 'degree': [1, 3, 5, 7]}

estimator = SVC(class_weight=class_weights)

estimator_random = RandomizedSearchCV(estimator=estimator, param_distributions=tuned_parameters, n_iter=40,
                    n_jobs=-1, cv=3, verbose=2, scoring='balanced_accuracy', error_score=np.nan)

estimator_random.fit(X_train, y_train_int)

print("Best Parameters:")
print()
print(estimator_random.best_params_)
print()
means = estimator_random.cv_results_['mean_test_score']
stds = estimator_random.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, estimator_random.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()

res = estimator_random.cv_results_
np.save('parameters.npy', res)


### Output ###

"""

Best Parameters:

{'kernel': 'rbf', 'gamma': 'auto', 'degree': 1, 'C': 10}

0.403 (+/-0.004) for {'kernel': 'rbf', 'gamma': 'auto', 'degree': 5, 'C': 1}
0.417 (+/-0.012) for {'kernel': 'rbf', 'gamma': 0.001, 'degree': 3, 'C': 50}
0.438 (+/-0.013) for {'kernel': 'rbf', 'gamma': 0.01, 'degree': 7, 'C': 10}
0.374 (+/-0.005) for {'kernel': 'rbf', 'gamma': 0.001, 'degree': 3, 'C': 5}
0.369 (+/-0.011) for {'kernel': 'rbf', 'gamma': 0.1, 'degree': 7, 'C': 50}
0.433 (+/-0.010) for {'kernel': 'rbf', 'gamma': 0.01, 'degree': 3, 'C': 5}
0.364 (+/-0.009) for {'kernel': 'rbf', 'gamma': 0.1, 'degree': 3, 'C': 100}
0.394 (+/-0.015) for {'kernel': 'poly', 'gamma': 'auto', 'degree': 1, 'C': 5}
0.359 (+/-0.023) for {'kernel': 'poly', 'gamma': 'auto', 'degree': 5, 'C': 1}
0.359 (+/-0.014) for {'kernel': 'poly', 'gamma': 0.001, 'degree': 1, 'C': 10}
0.405 (+/-0.009) for {'kernel': 'poly', 'gamma': 0.1, 'degree': 1, 'C': 10}
0.332 (+/-0.013) for {'kernel': 'rbf', 'gamma': 0.001, 'degree': 7, 'C': 1}
0.391 (+/-0.012) for {'kernel': 'poly', 'gamma': 0.01, 'degree': 3, 'C': 5}
0.413 (+/-0.010) for {'kernel': 'rbf', 'gamma': 0.1, 'degree': 3, 'C': 1}
0.329 (+/-0.022) for {'kernel': 'poly', 'gamma': 0.01, 'degree': 7, 'C': 50}
0.404 (+/-0.009) for {'kernel': 'poly', 'gamma': 'auto', 'degree': 3, 'C': 100}
0.433 (+/-0.010) for {'kernel': 'rbf', 'gamma': 0.01, 'degree': 7, 'C': 5}
0.410 (+/-0.020) for {'kernel': 'poly', 'gamma': 0.01, 'degree': 3, 'C': 50}
0.435 (+/-0.017) for {'kernel': 'rbf', 'gamma': 0.01, 'degree': 5, 'C': 100}
0.440 (+/-0.017) for {'kernel': 'rbf', 'gamma': 'auto', 'degree': 1, 'C': 10}
0.402 (+/-0.012) for {'kernel': 'rbf', 'gamma': 0.01, 'degree': 7, 'C': 1}
0.400 (+/-0.015) for {'kernel': 'poly', 'gamma': 0.001, 'degree': 1, 'C': 100}
0.395 (+/-0.013) for {'kernel': 'rbf', 'gamma': 0.1, 'degree': 5, 'C': 10}
0.404 (+/-0.009) for {'kernel': 'poly', 'gamma': 0.01, 'degree': 1, 'C': 50}
0.402 (+/-0.012) for {'kernel': 'rbf', 'gamma': 0.01, 'degree': 3, 'C': 1}
0.437 (+/-0.013) for {'kernel': 'rbf', 'gamma': 'auto', 'degree': 7, 'C': 5}
0.440 (+/-0.017) for {'kernel': 'rbf', 'gamma': 'auto', 'degree': 7, 'C': 10}
0.437 (+/-0.013) for {'kernel': 'rbf', 'gamma': 'auto', 'degree': 3, 'C': 5}
0.403 (+/-0.004) for {'kernel': 'rbf', 'gamma': 'auto', 'degree': 7, 'C': 1}
0.423 (+/-0.003) for {'kernel': 'rbf', 'gamma': 0.001, 'degree': 7, 'C': 100}
0.404 (+/-0.018) for {'kernel': 'poly', 'gamma': 0.01, 'degree': 3, 'C': 10}
0.341 (+/-0.019) for {'kernel': 'poly', 'gamma': 0.1, 'degree': 5, 'C': 10}
0.375 (+/-0.026) for {'kernel': 'poly', 'gamma': 'auto', 'degree': 5, 'C': 5}
0.157 (+/-0.007) for {'kernel': 'poly', 'gamma': 0.001, 'degree': 3, 'C': 5}
0.359 (+/-0.014) for {'kernel': 'poly', 'gamma': 0.01, 'degree': 1, 'C': 1}
0.126 (+/-0.000) for {'kernel': 'poly', 'gamma': 0.001, 'degree': 7, 'C': 50}
0.438 (+/-0.013) for {'kernel': 'rbf', 'gamma': 0.01, 'degree': 1, 'C': 10}
0.369 (+/-0.016) for {'kernel': 'poly', 'gamma': 'auto', 'degree': 7, 'C': 100}
0.413 (+/-0.010) for {'kernel': 'rbf', 'gamma': 0.1, 'degree': 5, 'C': 1}
0.407 (+/-0.022) for {'kernel': 'poly', 'gamma': 0.1, 'degree': 1, 'C': 50}


"""