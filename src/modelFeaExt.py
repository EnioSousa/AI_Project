from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pandas as pd
import sys
import numpy as np
import info

def linearModel(args):
    trainFile = info.dataDir + "description/"
    testFile = trainFile

    if ( args.pandas):
        trainFile += "trainPandaDesc.csv"
        testFile += "testPandaDesc.csv"

    else:
        trainFile += "trainDesc.csv"
        testFile += "testDesc.csv"

    dataTrain = pd.read_csv(trainFile)
    dataTest = pd.read_csv(testFile)

    trainValue = dataTrain[["ear_pointy", "nose_size"]].to_numpy()
    trainLabel = dataTrain[["animal_type"]].to_numpy().flatten()

    testValue = dataTest[["ear_pointy", "nose_size"]].to_numpy()
    testLabel = dataTest[["animal_type"]].to_numpy().flatten()

    # fit final model
    model = LogisticRegression()
    model.fit(trainValue, trainLabel)


    predictedLabel = model.predict(testValue)

    # show the inputs and predicted outputs
    count = 0
    wrong = 0
    for i in range(len(testValue)):
        print("X=%s, Predicted=%s" % (testLabel[i], predictedLabel[i]))

        if ( testLabel[i] != predictedLabel[i]):
            wrong += 1

        count += 1

    print("Acc %.2f" % ((count - wrong) / count))