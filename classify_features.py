"""classify_features.py: 
This script contains the code for feature classification (task 1) of the ISM project.
"""
__author__      = "Sebastian Engelhardt"

# some standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
# prepocessing tools
from sklearn.preprocessing import StandardScaler
# classifiers
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
# evaluation tools
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from scikitplot.metrics import plot_roc, plot_confusion_matrix
# tool to save the model
from joblib import dump, load
# custom functions
from extract_val_data import extract_val_data

class Classifier():
    # pylint: disable=too-many-instance-attributes
    # it's just convenient to store the different datasets as class attributes
    # pylint: disable=attribute-defined-outside-init
    # we define the dataset variables in a function called in init to keep the code clean
    def __init__(self, algorithm="svm", dataset="features/features_and_label.csv", test_dataset=None):
        self.algorithm = algorithm
        if algorithm == "svm":
            self.estimator = SVC(C=10, kernel = 'rbf', gamma='auto', probability=True, class_weight="balanced", break_ties=True, decision_function_shape='ovr')
        elif algorithm == "bagging_svm":
            self.estimator = BaggingClassifier(SVC(C=3, kernel = 'rbf', gamma='auto', probability=True, class_weight="balanced"),
                            n_jobs=-1, verbose=1)
        elif algorithm == "calib_svm":
            self.estimator = CalibratedClassifierCV(SVC(C=10, kernel = 'rbf', gamma='auto', probability=True, class_weight="balanced", break_ties=True, decision_function_shape='ovr'))
        elif algorithm == "tree":
            self.estimator = DecisionTreeClassifier(max_features=10, class_weight="balanced")
        elif algorithm == "forest":
            self.estimator = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                            min_samples_leaf=6, class_weight="balanced", n_jobs=-1)
        elif algorithm == "bagging_tree":
            self.estimator = BaggingClassifier(DecisionTreeClassifier(class_weight="balanced"), n_jobs=-1, n_estimators=30)
        elif algorithm == "knn":
            self.estimator = KNeighborsClassifier(n_neighbors=20, n_jobs=-1, weights="distance")
        elif algorithm == "ada_boost":
            base = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                    min_samples_leaf=6, class_weight="balanced", n_jobs=-1)
            self.estimator = AdaBoostClassifier(base_estimator=base)
        else:
            warnings.warn("unknown option "+str(algorithm)+", load a joblib file or quit", SyntaxWarning)
            self.estimator = None # has to be loaded from file
        
        self._prepare_dataset(dataset, test_dataset)
        self.y_pred = []
        self.y_test = []
    
    def _prepare_dataset(self, dataset, test_dataset):
        """ Load Dataset, extract features and labels, scale the values
        Function is called in init, you don't need to call it from outside
        """
        # read data
        train_d, val_d = extract_val_data(dataset)
        # extract features and labels
        # the last 9 elements of the dataset are the (one hot) labels
        X_train = train_d.iloc[:,1:-9].values
        X_val = val_d.iloc[:,1:-9].values
        y_train = train_d.iloc[:,-9:].values
        y_val = val_d.iloc[:,-9:].values
        self.val_names = val_d.iloc[:,0].values
        print("using ", X_train.shape[1], " features")
        # scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_val = scaler.transform(X_val)
        # convert label from one-hot to index (needed by some functions)
        self.y_train = np.argmax(y_train, axis=1)
        self.y_val = np.argmax(y_val, axis=1)
        if test_dataset is not None:
            test_d = pd.read_csv(test_dataset)
            X_test = test_d.iloc[:,1:].values
            self.X_test = scaler.transform(X_test)
            self.test_names = test_d.iloc[:,0].values

    def store_classifier(self, filename=""):
        """ store classifier to file
        """
        if filename == "":
            filename = "estimator_"+self.algorithm+".joblib"
        dump(self.estimator, filename)

    def load_classifier(self, filename):
        """ load trained classifier
        """
        try:
            self.estimator=load(filename)
        except FileNotFoundError:
            # exit programm, we don't have a fitted classifier
            raise SystemExit("Could not load estimator, please train one")
    
    def train(self):
        """ fit the classifier to the traing data
        """
        print("traing started, this may take a while")
        self.estimator.fit(self.X_train, self.y_train)

    def classify(self, dataset_type="val"):
        """ predict labels for test- or validation data
        """
        if dataset_type=="val":
            self.y_pred = self.estimator.predict(self.X_val)
            return self.y_pred
        if dataset_type=="test":
            self.y_test = self.estimator.predict(self.X_test)
            return self.y_test
        # else case, option was not val or test
        raise ValueError("unknown option "+str(dataset_type)+" valid options are 'val' and 'test'")

    def classify_with_threshold(self, threshold=0.0, dataset_type="val"):
        """ predict probabilities for labels of testdata.
        SVM does not support a consistent probability estimate, the value of the decision function is used in that case.
        The threshold is not a probability in that case!
        """
        # what dataset are we working with?
        if dataset_type == "val":
            X = self.X_val
        elif dataset_type == "test":
            X = self.X_test
        else:
            raise ValueError("unknown option "+str(dataset_type)+" valid options are 'val' and 'test'")
        # do we have the special case of a SVM?
        if self.algorithm=="svm":
            probs=self.estimator.decision_function(X)
        else:
            assert threshold <=1 and threshold >=0, "threshold has to be in [0,1]"
            probs=self.estimator.predict_proba(X)
        # apply threshold and write result
        pred = self.__threshold_probs(threshold, probs)
        if dataset_type == "val":
            self.y_pred = pred
        else:   # will be test because of the check above
            self.y_test = pred
        return pred

    @staticmethod
    def __threshold_probs(threshold, probs):
        """ Helper function for `classify_with_threshold`
        """
        pred = []
        for sample in probs:
            max_val = np.max(sample)
            if max_val >= threshold:
                pred.append(np.argmax(sample))
            else:
                pred.append(8)
        return np.asarray(pred)

    def print_evaluation(self):
        """ print some evaluation metrices
        """
        print("#### Evaluation #### \n")
        print("confusion matrix:")
        print(confusion_matrix(self.y_val,self.y_pred))
        print("classification report: ")
        print(classification_report(self.y_val,self.y_pred))

        print("balanced_accuracy_score: ")
        print(balanced_accuracy_score(self.y_val,self.y_pred))

    def show_roc(self):
        """ create ROC curves
        """
        # requires probas, otherwise not enough thresholds
        pred = self.estimator.predict_proba(self.X_val)
        plot_roc(self.y_val, pred)
        plt.show()
    
    def show_confusion_matrix(self):
        """ create a heat map showing the confusion matrix
        """
        # version from scikitplot doesn't show labels as text but looks better otherwise
        ax = plot_confusion_matrix(self.y_val, self.y_pred, normalize=True)
        ax.set_ylim(7.5 ,-0.5) # fix layout
        plt.show()
    
    def compare_confidence(self, only_correct=False):
        """ This function can be used to find a confidence threshold.
        It shows an histogram of the distribution of class prediction confidence for 
        test- and validation-dataset. 
        """
        if self.algorithm=="svm":
            val=self.estimator.decision_function(self.X_val)
            test=self.estimator.decision_function(self.X_test)
        else:
            val=self.estimator.predict_proba(self.X_val)
            test=self.estimator.predict_proba(self.X_test)
        # extrct correct predictions for val    
        val_right = []
        for i in range(len(val)):
            if np.argmax(val[i]) == self.y_val[i]:
                val_right.append(np.max(val[i]))
        val_right=np.asarray(val_right)

        max_val = np.max(val, axis=1)
        max_test = np.max(test, axis=1)
        # weigh to get fractions because they don't have the same number of samples
        if only_correct:
            plt.hist(val_right, alpha=0.5, label="val correct", weights=np.zeros_like(val_right) + 1. / val_right.size)
        else:
            plt.hist(max_val, alpha=0.5, label="val", weights=np.zeros_like(max_val) + 1. / max_val.size)
        plt.hist(max_test, alpha=0.5, label="test", weights=np.zeros_like(max_test) + 1. / max_test.size)
        plt.title("Confidence of class prediction \nfor validation- and test-dataset")
        if self.algorithm=="svm":
            plt.xlabel("decision function value")
        else:
            plt.xlabel("class probability")
        plt.ylabel("frequency")
        plt.legend(loc="best")
        plt.show()

    def write_classification(self, dataset_type="val"):
        """ write result to a file that can be uploaded
        """
        if dataset_type == "val":
            # convert index to one-hot
            pred=np.eye(9)[self.y_pred]
            names=self.val_names.reshape(-1,1) # required for concatenate
        elif dataset_type == "test":
            pred=np.eye(9)[self.y_test]
            names=self.test_names.reshape(-1,1)
        else:
            raise ValueError("unknown option "+str(dataset_type)+" valid options are 'val' and 'test'")

        # add filenames to predictions
        table = np.concatenate((names, pred), axis=1)
        # save with unique filename
        d = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        np.savetxt("output/classification_"+dataset_type+"_"+d+".csv", table, fmt='%s, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f',
                    delimiter=",", header="image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK", comments="")
        print("file was saved")

if __name__ == "__main__":
    classifier = Classifier(algorithm="svm", dataset="features/features_Kmeans_metadata_and_label_onehot.csv", 
                test_dataset="features/features_Kmeans_meta_test.csv")
    classifier.train()
    # classifier.store_classifier("estimator_svm_segment_km_meta.joblib")
    # classifier.load_classifier("estimator_svm_segment_km_meta.joblib")
    classifier.classify()
    classifier.print_evaluation()
    # classifier.show_roc()
    classifier.show_confusion_matrix()
    # classifier.compare_confidence(only_correct=True)
    # classifier.classify_with_threshold(7, dataset_type="test")
    # plt.hist(classifier.y_test, bins=9, align='left')
    # plt.show()
    # classifier.write_classification(dataset_type="test")