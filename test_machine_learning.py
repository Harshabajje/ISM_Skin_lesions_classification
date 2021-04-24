# pylint: disable=unused-import

# some standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# prepocessing tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
# classifiers
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# evaluation tools
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, balanced_accuracy_score
from scipy import interp
# tool to save the model
from joblib import dump

# load dataset using pandas
dataset = pd.read_csv("features/features_and_label.csv")
# print(dataset.head())

# seperate features and labels
X = dataset.iloc[:,1:71].values
y = dataset.iloc[:,71:81].values
print("using ", X.shape[1], " features")

# extract test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)    # TODO real split

# scale features down
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# compute label as int from one-hot
y_train_int = np.argmax(y_train, axis=1)
y_test_int = np.argmax(y_test, axis=1)

# comute class weights 
weights = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)
class_weights = dict(enumerate(weights))
print("class weights: ", class_weights)
# selecting a machinelearning model

# estimator = BaggingClassifier(SVC(C=3, kernel = 'rbf', gamma='auto', probability=True, class_weight=class_weights), n_jobs=-1, verbose=1) # very slow
estimator = SVC(C=10, kernel = 'rbf', gamma='auto', probability=True, class_weight=class_weights) # this is a bit slow
# estimator = RandomForestClassifier(n_estimators=100, class_weight=class_weights, n_jobs=-1)
# estimator = DecisionTreeClassifier(max_features=10, class_weight=class_weights)

# train model and predict test data
estimator.fit(X_train, y_train_int)
y_pred=estimator.predict(X_test)

dump(estimator, "estimator_dump")

# print some metrices
print("#### Evaluation #### \n")

print("confusion matrix:")
print(confusion_matrix(y_test_int,y_pred))
print("classification report: ")
print(classification_report(y_test_int,y_pred))

print("balanced_accuracy_score: ")
print(balanced_accuracy_score(y_test_int,y_pred))

# create ROC curve
# see https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

n_classes=8 # TODO adapt when we use unknown category 
fpr = [None]*n_classes  
tpr = [None]*n_classes
roc_auc = [None]*n_classes

for i in range(n_classes):
    fpr[i], tpr[i], __ = roc_curve(y_test_int, y_pred, i)
    roc_auc[i] = auc(fpr[i], tpr[i])

print("average area under curve")
print(np.average(roc_auc))

# compute average roc curve
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes
roc_auc_mean = auc(all_fpr, mean_tpr)

# plot the curves
plt.plot(all_fpr, mean_tpr, label='average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_mean), linestyle=':', linewidth=4)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
         ''.format(i, roc_auc[i]))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves')
plt.legend(loc="lower right")
plt.show()