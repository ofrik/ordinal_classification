__author__ = 'Ofri'

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import random
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datetime import datetime
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
        self.classifier = clone(classifier)
        self.orderedClassList = orderedClassList
        self.models = []

    def fit(self, X, y):
        """
        fit the ordinal classifier
        :param X: training datas
        :param y: training labels
        :return: the fitted model
        """
        if len(y) != len(X):
            raise "must be in the same length"
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
                    p = self.models[k - 1].predict_proba(x)[0][1]
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
        if len(y) != len(X):
            raise "must be in the same length"
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


def regressionToOrdinal(y):
    maxValue = max(y)
    minValue = min(y)
    part = (maxValue - minValue) / 3
    lowThreshold = minValue + part
    highThreshold = maxValue - part

    def makeOrdinal(x):
        if x <= lowThreshold:
            return 1
        elif x > highThreshold:
            return 3
        else:
            return 2

    arrayChanger = np.vectorize(makeOrdinal)
    return pd.DataFrame(arrayChanger(y))


if __name__ == "__main__":
    winequality_df = pd.read_csv("datasets/winequality-white.csv", sep=";")
    winequality_df_Y = winequality_df["quality"]
    winequality_df_X = winequality_df.drop("quality", axis=1)

    energyoutput_df = pd.read_csv("datasets\\energyoutput.csv")
    energyoutput_df_Y = energyoutput_df["PE"]
    energyoutput_df_X = energyoutput_df.drop("PE", axis=1)

    facebook_comments_df = pd.read_csv(
        "datasets\\facebook_comments.csv",
        names=["f" + str(i) for i in range(54)])
    facebook_comments_df_X = facebook_comments_df[facebook_comments_df.columns[:-1]]
    facebook_comments_df_Y = facebook_comments_df[facebook_comments_df.columns[-1:]]

    popularity_df = pd.read_csv("datasets\\OnlineNewsPopularity.csv")
    popularity_df_X = popularity_df[popularity_df.columns[2:-1]]
    popularity_df_Y = popularity_df[popularity_df.columns[-1:]]

    blogdata_df = pd.read_csv("datasets\\blogData_train.csv",
                              names=["f" + str(i) for i in range(281)])
    blogdata_df_X = blogdata_df[blogdata_df.columns[:-1]]
    blogdata_df_Y = blogdata_df[blogdata_df.columns[-1:]]

    datasets = [("Wine Quality", winequality_df_X, regressionToOrdinal(winequality_df_Y.values)),
                ("Combined Cycle Power Plant", energyoutput_df_X, regressionToOrdinal(energyoutput_df_Y.values)),
                ("BlogFeedback", blogdata_df_X, regressionToOrdinal(blogdata_df_Y.values)),
                ("Facebook Comment Volume", facebook_comments_df_X, regressionToOrdinal(facebook_comments_df_Y.values)),
                ("Online News Popularity", popularity_df_X, regressionToOrdinal(popularity_df_Y.values))]

    for dataset in datasets:
        print "comparing %s" % (dataset[0])
        X = dataset[1]
        y = dataset[2]
        kf = KFold(n_splits=5)
        originalScores = []
        originalFitTime = []
        ordinalScores = []
        ordinalFitTime = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            estimator = SVC(probability=True)
            ordinal = OrdinalClassifier(estimator, [1, 2, 3])
            # print "fit original classifier"
            t0 = datetime.now()
            estimator.fit(X_train, y_train)
            originalFitTime.append(datetime.now() - t0)
            t0 = datetime.now()
            ordinal.fit(X_train, y_train)
            ordinalFitTime.append(datetime.now() - t0)
            originalScores.append(estimator.score(X_test, y_test))
            ordinalScores.append(ordinal.score(X_test, y_test))
        print "\noriginal mean score: %s\nordinal mean score: %s\n\n" % (
        np.mean(originalScores),np.mean(ordinalScores))


        # print regressionToOrdinal([random.randrange(0, 100) for i in range(1000)])
        # y = pd.DataFrame([random.randrange(0, 3) for i in range(1000)])
        # X = pd.DataFrame([[random.randrange(0, 10) for i in range(4)] for j in range(1000)])
        # estimator = SVC(probability=True)
        # estimator.fit(X, y)
        # print estimator.score(X, y)
        # classList = [0, 1, 2]
        # myclf = OrdinalClassifier(estimator, classList)
        # myclf.fit(X, y)
        # print myclf.score(X, y)
        # print "started"
