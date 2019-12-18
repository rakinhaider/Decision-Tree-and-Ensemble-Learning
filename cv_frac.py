import pandas as pd
import numpy as np
from scipy import stats
import utils as util
from trees import decision_tree, bagging, random_forests
import matplotlib.pyplot as plt


def split_data_into_folds(data, k):
    data_per_fold = int(len(data) / k)
    folds = []
    for i in range(k):
        folds.append(data[data_per_fold * i:data_per_fold * (i + 1)])
    return folds


def get_average_and_se(a, k):
    average = np.sum(a) / k
    deviations = np.array(a, dtype='float64') - average
    variance = deviations * deviations
    variance = variance / k
    sd = np.sqrt(np.sum(variance, dtype='float32'))
    se = sd / np.sqrt(k)
    return average, se


def read_model_results_from_file(model_name, tr_or_ts):
    f = open('outputs/' + model_name + '_' + tr_or_ts + '_ac.txt', 'r')
    avg_ac = []
    l = int(f.readline())
    for i in range(l):
        avg_ac.append(float(f.readline()))
    f.close()
    f = open('outputs/' + model_name + '_' + tr_or_ts + '_se.txt', 'r')
    se = []
    l = int(f.readline())
    for i in range(l):
        se.append(float(f.readline()))
    f.close()
    return avg_ac, se


def write_model_results_to_file(model_name, tr_or_ts, avg_ac, se):
    f = open('outputs/' + model_name + '_' + tr_or_ts + '_ac.txt', 'w')
    f.write(str(len(avg_ac)) + '\n')
    for ac in avg_ac:
        f.write(str(ac) + '\n')
    f.close()

    if len(se) > 0:
        f = open('outputs/' + model_name + '_' + tr_or_ts + '_se.txt', 'w')
        f.write(str(len(se)) + '\n')
        for e in se:
            f.write(str(e) + '\n')
        f.close()


def generate_folds(data, number_of_folds):
    data = data.sample(random_state=18, frac=1)
    folds = split_data_into_folds(data, number_of_folds)
    return folds


def get_model_function(model_idx):
    if model_idx == 1:
        return decision_tree
    elif model_idx == 2:
        return bagging
    elif model_idx == 3:
        return random_forests


def get_model_name(model_idx):
    if model_idx == 1:
        return 'DT'
    elif model_idx == 2:
        return 'BT'
    elif model_idx == 3:
        return 'RF'


def run_cross_validation_for_model(data, t_fracs, number_of_folds, model_idx):
    folds = generate_folds(data, number_of_folds)
    model_func = get_model_function(model_idx)
    avg_ac_tfrac = {'tr': [], 'ts': []}
    se_tfrac = {'tr': [], 'ts': []}
    print(get_model_name(model_idx).upper())
    for t_frac in t_fracs:
        fold_tr_ac = []
        fold_ts_ac = []
        print('t_frac:', t_frac)
        for i in range(number_of_folds):
            test_set = folds[i]
            train_set = pd.DataFrame()
            for j in range(number_of_folds):
                if j != i:
                    train_set = train_set.append(folds[j])
            train_set = train_set.sample(random_state=32, frac=t_frac)
            training_ac, test_ac = model_func(train_set, test_set)
            fold_tr_ac.append(training_ac)
            fold_ts_ac.append(test_ac)
            print('Fold:', i)
            print('Training Accuracy {}: {:.2f}'.format(get_model_name(model_idx), training_ac))
            print('Testing Accuracy {}: {:.2f}'.format(get_model_name(model_idx), test_ac))

        write_model_results_to_file('frac_'+ get_model_name(model_idx) + '_' + str(t_frac), 'tr', fold_tr_ac, [])
        write_model_results_to_file('frac_'+get_model_name(model_idx) + '_' + str(t_frac), 'ts', fold_ts_ac, [])

        avg_ac, se = get_average_and_se(fold_tr_ac, number_of_folds)
        avg_ac_tfrac['tr'].append(avg_ac)
        se_tfrac['tr'].append(se)
        #print('Average Training Accuracy {}: {:.2f}'.format(get_model_name(model_idx), avg_ac))
        #print('Standard Error {}: {:.2f}'.format(get_model_name(model_idx), se))

        avg_ac, se = get_average_and_se(fold_ts_ac, number_of_folds)
        avg_ac_tfrac['ts'].append(avg_ac)
        se_tfrac['ts'].append(se)
        print('Average Testing Accuracy {}: {:.2f}'.format(get_model_name(model_idx), avg_ac))
        print('Standard Error {}: {:.2f}'.format(get_model_name(model_idx), se))

    return avg_ac_tfrac, se_tfrac


def read_results_for_frac(model_name, train_or_test, frac):
    f = open(model_name + '_' + str(frac) + '_' + train_or_test + '_ac.txt', 'r')
    l = int(f.readline())
    acs = []
    for i in range(l):
       acs.append(float(f.readline()))

    return acs


def test_hypothesis(a_results, b_results):
    t, p_value = stats.ttest_rel(a_results, b_results)
    print('t:', t)
    print('p_value:', p_value)


if __name__ == "__main__":
    if util.final:
        columns, data = util.readFile('trainingSet.csv')
    else:
        columns, data = util.readFile('test_trainingSet.csv')

    number_of_folds = 10

    folds = generate_folds(data, number_of_folds)

    # Answer to the question 4
    t_fracs = [0.05, 0.075, 0.1, 0.15, 0.2]
    fold_size = len(folds[0])
    train_sizes = [fold_size * (number_of_folds - 1) * frac for frac in t_fracs]
    # model_idxs = [1, 2, 3]
    model_idxs = []

    for model_idx in model_idxs:
        avg_ac_tfrac, se_tfrac = run_cross_validation_for_model(data, t_fracs, number_of_folds, model_idx)
        for x in avg_ac_tfrac:
            write_model_results_to_file('frac_'+ get_model_name(model_idx), x, avg_ac_tfrac[x], se_tfrac[x])

    # Answer to the question 4.a
    model_idxs = [1, 2, 3]
    for model_idx in model_idxs:
        sets = ['ts']
        for s in sets:
            avg_ac, se = read_model_results_from_file('frac_' + get_model_name(model_idx), s)
            plt.errorbar(train_sizes, avg_ac, yerr=se, label=get_model_name(model_idx) + '_' + s + '_ac', fmt='o-')

    plt.legend(loc='upper right')
    plt.xticks(train_sizes)
    plt.tight_layout(pad=3)
    plt.xlabel('Samples used in training.')
    plt.ylabel('Model Accuracy')
    plt.savefig('outputs/' + '4_a.pdf', format='pdf')
    plt.clf()

    # Answer to the question 4.b
    test_hypothesis(read_results_for_frac('outputs/frac_' + get_model_name(2), 'ts', 0.15),
                    read_results_for_frac('outputs/frac_' + get_model_name(1), 'ts', 0.15)
                    )
