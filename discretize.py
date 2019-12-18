import pandas as pd
import numpy as np
import utils as util
import sys


def makeRanges(data, continuousValuedColumns):
    startValues = {}
    endValues = {}
    for col in continuousValuedColumns:
        startValues[col] = 0
        endValues[col] = max(10, data[col].max())

    # Handle age
    startValues['age'] = 18
    startValues['age_o'] = 18
    endValues['age'] = max(58, data['age'].max())
    endValues['age_o'] = max(58, data['age_o'].max())

    for col in util.psParticipants:
        startValues[col] = 0
        endValues[col] = 1
    for col in util.psPartners:
        startValues[col] = 0
        endValues[col] = 1

    # Handle correlation
    startValues['interests_correlate'] = -1
    endValues['interests_correlate'] = max(1, data['interests_correlate'].max())

    return startValues, endValues


def continuousToBinConverter(data, columns, binCount):
    continuousValuedColumns = list(columns)
    for col in util.notContinuousValuedColumns:
        if col in continuousValuedColumns:
            continuousValuedColumns.remove(col)

    startValues, endValues = makeRanges(data, continuousValuedColumns)

    for col in continuousValuedColumns:
        ranges = endValues[col] - startValues[col]
        binSize = ranges / binCount

        bins = [x for x in np.arange(startValues[col], endValues[col] + binSize, binSize)]
        labels = [x for x in np.arange(0, len(bins) - 1, 1)]
        binnedColumn = pd.cut(data[col], bins=bins, labels=labels, include_lowest=True)
        data = data.drop([col], axis=1)
        data.insert(columns.index(col), col, binnedColumn)
        columns = list(data.columns)

    return data, continuousValuedColumns


if __name__ == "__main__":
    if util.final:
        columns, data = util.readFile(sys.argv[1])
    else:
        columns, data = util.readFile('test_dating.csv')

    data, continuous_columns = continuousToBinConverter(data, columns, 5)

    for col in continuous_columns:
        print(col + ': ', end='')
        print(data[col].value_counts().sort_index().tolist())

    if util.final:
        data.to_csv(sys.argv[2])
    else:
        data.to_csv('test_dating-binned.csv')
