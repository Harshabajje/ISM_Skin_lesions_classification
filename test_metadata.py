import numpy as np
from extract_test_data import extract_test_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import datetime

dataset_m = "features/metadata_and_label_onehot.csv"
train_dm, test_dm = extract_test_data(dataset_m)

# extract features and labels
X_train_m = train_dm.iloc[:,1:32].values
X_test_m = test_dm.iloc[:,1:32].values
y_train = train_dm.iloc[:,32:41].values
y_test = test_dm.iloc[:,32:41].values

dataset = "features/features_Kmeans_Segmentation.csv"

train_d, test_d = extract_test_data(dataset)

X_train_f = train_d.iloc[:,1:71].values
X_test_f = test_d.iloc[:,1:71].values

scaler = StandardScaler()
X_train_f = scaler.fit_transform(X_train_f)
X_test_f = scaler.transform(X_test_f)

X_train = np.hstack((X_train_f, X_train_m))
X_test = np.hstack((X_test_f, X_test_m))

print(X_train.shape)

y_train_int = np.argmax(y_train, axis=1)
y_test_int = np.argmax(y_test, axis=1)

# estimator = SVC(C=10, kernel = 'rbf', gamma='auto', probability=True, class_weight="balanced")
# estimator.fit(X_train, y_train_int)
# dump(estimator, "features_and_data.joblib")

estimator = load("features_and_data.joblib")

y_pred = estimator.predict(X_test)

print("confusion matrix:")
print(confusion_matrix(y_test_int,y_pred))
print("classification report: ")
print(classification_report(y_test_int,y_pred))

print("balanced_accuracy_score: ")
print(balanced_accuracy_score(y_test_int,y_pred))

test_names = test_d.iloc[:,0].values

pred=np.eye(9)[y_pred]
# add filenames to predictions
test_names=test_names.reshape(-1,1) 
table = np.concatenate((test_names, pred), axis=1)
# save with unique filename
d = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
np.savetxt("output/classification_"+d+".csv", table, fmt='%s, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f',
            delimiter=",", header="image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK", comments="")