import pandas as pd
import numpy as np
import utils as util
import sys
from decision_tree import DecisionTree
from ensemble import Bagging, RandomForests


def decision_tree(train, test, depth=8):
    model = DecisionTree()
    model.fit(train, depth=depth)
    return model.get_accuracy(train), model.get_accuracy(test)


def bagging(trainingSet, testSet, m=30, depth=8):
    model = Bagging(base_classifier=DecisionTree)
    model.fit(trainingSet, m=m, depth=depth)
    return model.get_accuracy(trainingSet), model.get_accuracy(testSet)


def random_forests(trainingSet, testSet, m=30, depth=8):
    model = RandomForests()
    model.fit(trainingSet, m=m, depth=depth)
    return model.get_accuracy(trainingSet), model.get_accuracy(testSet)


def get_model_name(model_idx):
    if modelIdx == 1:
        return 'DT'
    elif modelIdx == 2:
        return 'BT'
    elif modelIdx == 3:
        return 'RF'


if __name__ == "__main__":
    training_data_filename = 'trainingSet.csv'
    test_data_filename = 'testSet.csv'
    modelIdx = 1

    if len(sys.argv) > 1:
        training_data_filename = sys.argv[1]
        test_data_filename = sys.argv[2]
        modelIdx = int(sys.argv[3])

    if util.final == True:
        columns, train = util.readFile(training_data_filename)
        _, test = util.readFile(test_data_filename)
    else:
        columns, train = util.readFile('test_' + training_data_filename)
        _, test = util.readFile('test_' + test_data_filename)

        columns = columns[:5] + [columns[-1]]
        train = train[columns]
        test = test[columns]

    print(train.head())

    if modelIdx == 1:
        training_ac, test_ac = decision_tree(train, test)
    elif modelIdx == 2:
        training_ac, test_ac = bagging(train, test)
    elif modelIdx == 3:
        training_ac, test_ac = random_forests(train, test)

    print('Training Accuracy {}: {:.2f}'.format(get_model_name(modelIdx), training_ac))
    print('Testing Accuracy {}: {:.2f}'.format(get_model_name(modelIdx), test_ac))
