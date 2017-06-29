__author__ = 'Ofri'

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
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
    facebook_comments_df = facebook_comments_df[:20000]
    facebook_comments_df_X = facebook_comments_df[facebook_comments_df.columns[:-1]]
    facebook_comments_df_Y = facebook_comments_df[facebook_comments_df.columns[-1:]]

    popularity_df = pd.read_csv("datasets\\OnlineNewsPopularity.csv")
    popularity_df = popularity_df[:10000]
    popularity_df_X = popularity_df[popularity_df.columns[2:-1]]
    popularity_df_Y = popularity_df[popularity_df.columns[-1:]]

    blogdata_df = pd.read_csv("datasets\\blogData_train.csv",
                              names=["f" + str(i) for i in range(281)])
    blogdata_df = blogdata_df[:5000]
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
            originalFitTime = []
            originalPredictTime = []
            originalAcc = []
            originalRecall = []
            originalPrecision = []
            originalF1 = []
            ordinalFitTime = []
            ordinalPredictTime = []
            ordinalAcc = []
            ordinalRecall = []
            ordinalPrecision = []
            ordinalF1 = []

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
                estimator_preds = estimator.predict(X_test)
                originalPredictTime.append((datetime.now() - t0).total_seconds())
                originalAcc.append(accuracy_score(y_test, estimator_preds))
                originalRecall.append(recall_score(y_test, estimator_preds))
                originalPrecision.append(precision_score(y_test, estimator_preds))
                originalF1.append(f1_score(y_test, estimator_preds))
                t0 = datetime.now()
                ordinal_preds = ordinal.predict(X_test)
                ordinalPredictTime.append((datetime.now() - t0).total_seconds())
                ordinalAcc.append(accuracy_score(y_test, ordinal_preds))
                ordinalRecall.append(recall_score(y_test, ordinal_preds))
                ordinalPrecision.append(precision_score(y_test, ordinal_preds))
                ordinalF1.append(f1_score(y_test, ordinal_preds))
            print "Original:\nFit time: %s\nPredictTime: %s\nAccuracy: %s\nRecall: %s\nPrecision: %s\nF1: %s\n\n" % (
                np.mean(originalFitTime), np.mean(originalPredictTime), np.mean(originalAcc), np.mean(originalRecall),
                np.mean(originalPrecision), np.mean(originalF1))
            print "Ordinal:\nFit time: %s\nPredictTime: %s\nAccuracy: %s\nRecall: %s\nPrecision: %s\nF1: %s\n\n" % (
                np.mean(ordinalFitTime), np.mean(ordinalPredictTime), np.mean(ordinalAcc), np.mean(ordinalRecall),
                np.mean(ordinalPrecision), np.mean(ordinalF1))
            print "the entire thing took %s seconds\n\n" % ((datetime.now() - t).total_seconds())


    for clf in clfs:
        checkClassifier(clf)
