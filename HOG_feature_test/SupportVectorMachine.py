from sklearn.svm import LinearSVC
from skimage.feature import hog
from skimage import data, exposure
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import pickle
from joblib import dump, load
from sklearn.model_selection import cross_val_score
import numpy as np

X = pickle.load(open('HOG_Features_25000_Images.dat', 'rb'))
X_Data=[]
length=[]
for i,item in enumerate(X):
    length.append(len(item))

max_length=max(length)

for i,item in enumerate(X):
    X_Data.append(np.pad(item, (0, max_length - len(item)), 'constant'))


y = pickle.load(open('HOG_Labels_25000_Images.dat', 'rb'))


Y=np.argmax(y,axis=1)

X_scaler = StandardScaler().fit(X_Data)
scaled_X = X_scaler.transform(X_Data)
rand_state = 42
print('scaling is done')
X_rem, X_test, y_rem, y_test = model_selection.train_test_split(scaled_X, Y, test_size=0.2, random_state=rand_state)
svc = LinearSVC(random_state=rand_state,verbose=False,max_iter=2000)
svc.fit(X_rem, y_rem)
print("Fit is done")

filename_model = 'SVM_Dataset_1000.dat'
pickle.dump(svc, open(filename_model, 'wb'))

print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
dump(svc, 'SVM.joblib')



accuracies = cross_val_score(estimator = svc, X = X_rem, y = y_rem, cv = 5)
mean = accuracies.mean()
variance = accuracies.std()
print(mean)
print(variance)
