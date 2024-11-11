from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils.tools import get_instance_readmitted
import os
from utils.metrics import auc_score
import numpy as np

BASE_CLASSIFIER = {"LR": LogisticRegression(C=1, penalty="l2"),
                   "SVC": SVC(C=1, kernel="linear", probability=True),
                   "LDA": LinearDiscriminantAnalysis(),
                   "QDA": QuadraticDiscriminantAnalysis(),
                   "NB": GaussianNB(),
                   "DCT": DecisionTreeClassifier(criterion='gini', max_depth=100, min_samples_leaf=5),
                   "RFT": RandomForestClassifier(n_estimators=100, max_depth=10, criterion='gini')}

class Evaluation:
    def __init__(self, base, setting, instances, path, representation):
        self.base = BASE_CLASSIFIER[base]
        self.evaluation_name = setting
        self.base_name = base
        self.instances = instances
        self.y = get_instance_readmitted(path, instances)
        self.x = representation


    def evaluate(self, x_test, path, instances):
        print(">>>>>>>start training evaluation {} classifier on {}: >>>>>>>>>>>>>>>>>>>>>>>>>>".format(self.base_name, self.evaluation_name))
        X, y = [], []
        for key in self.instances:
            X.append(self.x[key])
            y.append(self.y[key])
        self.base.fit(X, y)
        X_test, y_test = [], []
        y_dict = get_instance_readmitted(path, instances)
        for key in instances:
            X_test.append(x_test[key])
            y_test.append(y_dict[key])
        print(np.array(y_test).shape)
        y_pred_proba = self.base.predict_proba(X_test)
        auc_score(y=y_test, y_score=y_pred_proba)
        