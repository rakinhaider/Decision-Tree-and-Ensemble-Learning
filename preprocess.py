import pandas as pd
import numpy as np
import utils as util


def stripQuotes(data, stripColumns):
    changeCount = 0
    for col in stripColumns:
        for i in data.index.values:
            if "'" in data[col][i]:
                data.loc[i, col] = data.loc[i, col].replace("'", "")
                changeCount = changeCount + 1

    #print("Quotes removed from {0} cells.".format(changeCount))


def toLowerCase(data, toLowerColumns):
    changeCount = 0
    for col in toLowerColumns:
        for i in data.index.values:
            if data[col][i].islower() == False:
                data.loc[i, col] = data.loc[i, col].lower()
                changeCount = changeCount + 1

    #print("Standardized {0} cells to lower case.".format(changeCount))


def encodeLabels(data, encodeColumns, printLabels):
    labelMapping = {}
    for col in encodeColumns:
        cells = data[col]
        uniqueValues = cells.unique()
        labelMapping[col] = dict(zip(np.sort(uniqueValues), range(len(uniqueValues))))
        # print(labelMapping[col])

    for index, row in data.iterrows():
        for col in encodeColumns:
            data.loc[index, col] = labelMapping[col][row[col]]

    for key in printLabels.keys():
        pass
        #print('Value assigned for {} in column {}: {}.'. format(printLabels[key], key, labelMapping[key].get(printLabels[key], -1)))


def normalizeColumns(data, psParticipants, psPartners):
    for index, row in data.iterrows():
        sum = 0
        for col in psParticipants:
            sum = sum + row[col]

        for col in psParticipants:
            data.loc[index, col] = row[col] / sum

        sum = 0
        for col in psPartners:
            sum = sum + row[col]

        for col in psPartners:
            data.loc[index, col] = row[col] / sum

    # print(data[psParticipants+psPartners])
    for col in psParticipants:
        #print('Mean of {}: {:.2f}'.format(col, data[col].mean()))
        pass
    for col in psPartners:
        #print('Mean of {}: {:.2f}'.format(col, data[col].mean()))
        pass

import sys

if __name__ == "__main__":
    if util.final:
        # For final result run the following line
        columns, data = util.readFile(sys.argv[1],None)
    else:
        # For testing purpose run the following line.
        columns, data = util.readFile('test_dataset.csv')

    # Answer to question 1.i
    stripQuotes(data, ['race', 'race_o', 'field'])

    # Answer to question 1.ii
    toLowerCase(data, ['field'])

    # Answer to question 1.iii
    printLabels = {'gender': 'male',
                   'race': 'European/Caucasian-American',
                   'race_o': 'Latino/Hispanic American',
                   'field': 'law'}
    encodeLabels(data, ['gender', 'race', 'race_o', 'field'], printLabels)

    # Answer to question 1.iv

    normalizeColumns(data, util.psParticipants, util.psPartners)



    if util.final:
        # Run for final version
        data.to_csv(sys.argv[2])
    else:
        data.to_csv('test_dating.csv')
