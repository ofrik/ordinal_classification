__author__ = 'Ofri'

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pandas as pd
import random
from sklearn.base import clone
import operator
import warnings
warnings.filterwarnings("ignore")


class OrdinalClassifier(object):
    def __init__(self, classifier, orderedClassList):
        """
        initiating the class with prediction classifier and the ordinal classes order
        :param classifier: classifier to perform the prediction
        :param orderedClassList: list of ordered ordinal classes
        """
        self.classifier = classifier
        self.orderedClassList = orderedClassList
        self.models = []

    def fit(self, X, y):
        """
        fit the ordinal classifier
        :param X: training datas
        :param y: training labels
        :return: the fitted model
        """
        datasets = self.generateDatasets(y)
        for dataset in datasets:
            tmpclf = clone(self.classifier)
            tmpclf.fit(X, dataset)
            self.models.append(tmpclf)
        return self

    def predict(self, X):
        """
        predict the given data
        :param X: test data
        :return: list of predictions of the given data
        """
        predictions = []
        for x in X.values:
            index, bestProb = -1, -1
            for k in range(len(self.orderedClassList)):
                if k == 0:
                    p = self.models[k].predict_proba(x)[0][0]
                elif k == len(self.orderedClassList) - 1:
                    p = self.models[k-1].predict_proba(x)[0][1]
                else:
                    p = self.models[k - 1].predict_proba(x)[0][1] - self.models[k].predict_proba(x)[0][1]
                if p > bestProb:
                    index = k
                    bestProb = p
            predictions.append(self.orderedClassList[index])
        return predictions

    def score(self, X, y):
        """
        calculate the accuracy of the model on the given data
        :param X: test data
        :param y: test labels
        :return: the accuracy of the model on the given data
        """
        predictions = self.predict(X)
        correct = 0
        for i in range(len(y)):
            if predictions[i] == y.values[i][0]:
                correct += 1
        return float(correct) / len(y)

    def generateDatasets(self, y):
        """
        create list of different labels according the paper algorithm
        :param y: original labels
        :return: list of labels for each dataset as in the paper
        """
        datasets = []
        for i, c1 in enumerate(self.orderedClassList[:-1]):
            classes = []
            for c2 in self.orderedClassList[:i + 1]:
                classes.append(c2)
            def f(x):
                return 0 if x in classes else 1
            newds = y.copy(True)[0].apply(f)
            datasets.append(newds)
        return datasets


if __name__ == "__main__":
    y = pd.DataFrame([random.randrange(0, 3) for i in range(1000)])
    X = pd.DataFrame([[random.randrange(0, 10) for i in range(4)] for j in range(1000)])
    estimator = SVC(probability=True)
    estimator.fit(X, y)
    print estimator.score(X, y)
    classList = [0, 1, 2]
    myclf = OrdinalClassifier(estimator, classList)
    myclf.fit(X, y)
    print myclf.score(X, y)
    print "started"
