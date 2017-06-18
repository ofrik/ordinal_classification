__author__ = 'Ofri'

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.base import clone
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from datetime import datetime
from OrdinalClassifier import OrdinalClassifier
import warnings

warnings.filterwarnings("ignore")


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

    kf = KFold(n_splits=5)
    clfs = [("SVC", SVC(probability=True)), ("RandomForestClassifier", RandomForestClassifier(n_jobs=-1)),
            ("MLPClassifier", MLPClassifier())]


    def checkClassifier(clf):
        for dataset in datasets:
            t = datetime.now()
            print "comparing %s using %s" % (dataset[0], clf[0])
            X = dataset[1]
            print "has %s train records" % X.shape[0]
            y = dataset[2]
            originalScores = []
            originalFitTime = []
            originalScoreTime = []
            ordinalScores = []
            ordinalFitTime = []
            ordinalScoreTime = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                estimator = clf[1]
                ordinal = OrdinalClassifier(estimator, [1, 2, 3])
                t0 = datetime.now()
                estimator.fit(X_train, y_train)
                originalFitTime.append((datetime.now() - t0).total_seconds())
                t0 = datetime.now()
                ordinal.fit(X_train, y_train)
                ordinalFitTime.append((datetime.now() - t0).total_seconds())
                t0 = datetime.now()
                originalScores.append(estimator.score(X_test, y_test))
                originalScoreTime.append((datetime.now() - t0).total_seconds())
                t0 = datetime.now()
                ordinalScores.append(ordinal.score(X_test, y_test))
                ordinalScoreTime.append((datetime.now() - t0).total_seconds())
            print "original mean score: %s with mean time of %s seconds and mean score time of %s seconds\nordinal mean score: %s with mean time of %s seconds and mean score time of %s seconds" % (
                np.mean(originalScores), np.mean(originalFitTime), np.mean(originalScoreTime), np.mean(ordinalScores),
                np.mean(ordinalFitTime), np.mean(ordinalScoreTime))
            print "the entire thing took %s seconds\n\n" % ((datetime.now() - t).total_seconds())


    for clf in clfs:
        checkClassifier(clf)
