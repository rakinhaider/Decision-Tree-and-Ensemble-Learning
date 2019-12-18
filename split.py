import pandas as pd
import numpy as np
import utils as util


def split(data, random_state, frac):
    test = data.sample(random_state=random_state, frac=frac)
    train = data.drop(test.index)
    assert (len(train) + len(test) == len(data))
    return train, test


def save_train_and_test_split(train, test):
    if util.final:
        train.sort_index().to_csv('trainingSet.csv')
        test.sort_index().to_csv('testSet.csv')
    else:
        train.sort_index().to_csv('test_trainingSet.csv')
        test.sort_index().to_csv('test_testSet.csv')


if __name__ == '__main__':
    if util.final:
        columns, data = util.readFile('dating-binned.csv')
    else:
        columns, data = util.readFile('test_dating-binned.csv')

    train, test = split(data, 47, 0.2)

    save_train_and_test_split(train, test)
