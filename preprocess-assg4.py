import pandas as pd
import numpy as np
import preprocess as pr
import utils as util
import discretize as dscrt
import split as splt

if __name__=='__main__':
    if util.final:
        columns, data = util.readFile('dating-full.csv', index_col=None)
    else:
        column, data = util.readFile('test_dataset.csv')

    data = data[:6500]

    # Answer to the question 1.i
    data = data.drop(columns=['race', 'race_o', 'field'])
    # Answer to the question 1.ii
    pr.encodeLabels(data, ['gender'], {})

    # Answer to the question 1.iii
    pr.normalizeColumns(data, util.psParticipants, util.psPartners)

    # Answer to the question 1.iv
    data, _ = dscrt.continuousToBinConverter(data, list(data.columns), 2)

    # Answer to the question 1.v
    train, test = splt.split(data, 47, 0.2)
    splt.save_train_and_test_split(train, test)
