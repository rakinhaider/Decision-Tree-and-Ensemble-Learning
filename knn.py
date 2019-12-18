import pandas as pd
import numpy as np
import utils as util
from scipy import stats
import sys
import matplotlib.pyplot as plt
import importlib
xrs_validation = importlib.import_module("cv_frac")


class KNN:
    def __init__(self):
        self.data = None
        self.k = -1
        self.dist_function = None

    def fit(self, data, k, dist_function=1):
        self.data = data
        self.k = k
        if dist_function == 1:
            self.dist_function = self.euclidean_dist
        elif dist_function == 2:
            self.dist_function = self.manhattan_dist
        elif dist_function == 3:
            self.dist_function = self.cosine_dist

    def cosine_dist(self, x):
        dist = self.data.dot(x)
        # print(dist)
        # print(x)
        dist = dist[dist.columns[:-1]]
        # print(dist)
        dist = dist.sum(axis=1)
        # print(dist)
        return dist

    def manhattan_dist(self, x):
        dist = (self.data-x)
        # print(dist)
        # print(x)
        dist = dist[dist.columns[:-1]]
        # print(dist)
        dist = dist.sum(axis=1)
        # print(dist)
        return dist

    def euclidean_dist(self, x):
        dist = (self.data-x)**2
        # print(dist)
        dist = dist[dist.columns[:-1]]
        # print(dist)
        dist = dist.sum(axis=1)
        # print(np.sqrt(dist))
        return np.sqrt(dist)

    def predict(self, data):
        predictions = []
        for idx_x, x in data.iterrows():
            #print(idx_x)
            distances = self.dist_function(x)
            top_k = distances.sort_values(ascending=True).iloc[:self.k]
            #print(top_k)
            indices = top_k.index
            distances = top_k.values
            labels = self.data[data.columns[-1]][indices]
            # print(indices)
            # print(distances)
            # print(labels)
            mode, count = stats.mode(labels)
            #print('mode', mode[0], 'count', count)
            predictions.append(mode[0])
        return predictions

    def get_accuracy(self, test):
        predictions = self.predict(test)
        # print(predictions)
        target_attribute = test.columns[-1]
        # print(test[target_attribute])
        correct = (test[target_attribute] == predictions).astype(int).value_counts()[1]
        return correct * 100 / len(test)


def run_cross_validation_for_k(data, k_neigbours, number_of_folds):
    folds = xrs_validation.generate_folds(data, number_of_folds)
    model_func = knn
    avg_ac_tfrac = {'tr': [], 'ts': []}
    se_tfrac = {'tr': [], 'ts': []}
    print('KNN')
    for k in k_neigbours:
        fold_tr_ac = []
        fold_ts_ac = []
        print('k:', k)
        for i in range(number_of_folds):
            test_set = folds[i]
            train_set = pd.DataFrame()
            for j in range(number_of_folds):
                if j != i:
                    train_set = train_set.append(folds[j])
            training_ac, test_ac = model_func(train_set, test_set, int(k))
            fold_tr_ac.append(training_ac)
            fold_ts_ac.append(test_ac)
            print('Fold:', i)
            print('Training Accuracy {}: {:.2f}'.format('KNN', training_ac))
            print('Testing Accuracy {}: {:.2f}'.format('KNN', test_ac))

        xrs_validation.write_model_results_to_file('k_'+ 'KNN' + '_' + str(k), 'tr', fold_tr_ac, [])
        xrs_validation.write_model_results_to_file('k_'+ 'KNN' + '_' + str(k), 'ts', fold_ts_ac, [])

        avg_ac, se = xrs_validation.get_average_and_se(fold_tr_ac, number_of_folds)
        avg_ac_tfrac['tr'].append(avg_ac)
        se_tfrac['tr'].append(se)
        #print('Average Training Accuracy {}: {:.2f}'.format(get_model_name(model_idx), avg_ac))
        #print('Standard Error {}: {:.2f}'.format(get_model_name(model_idx), se))

        avg_ac, se = xrs_validation.get_average_and_se(fold_ts_ac, number_of_folds)
        avg_ac_tfrac['ts'].append(avg_ac)
        se_tfrac['ts'].append(se)
        print('Average Testing Accuracy {}: {:.2f}'.format('KNN', avg_ac))
        print('Standard Error {}: {:.2f}'.format('KNN', se))

    return avg_ac_tfrac, se_tfrac


def run_cross_validation_for_dist(data, dist_functions, number_of_folds):
    folds = xrs_validation.generate_folds(data, number_of_folds)
    model_func = knn
    avg_ac_tfrac = {'tr': [], 'ts': []}
    se_tfrac = {'tr': [], 'ts': []}
    print('KNN')
    for dist_function in dist_functions:
        fold_tr_ac = []
        fold_ts_ac = []
        print('k:', k)
        for i in range(number_of_folds):
            test_set = folds[i]
            train_set = pd.DataFrame()
            for j in range(number_of_folds):
                if j != i:
                    train_set = train_set.append(folds[j])
            training_ac, test_ac = model_func(train_set, test_set, dist_function=int(dist_functions))
            fold_tr_ac.append(training_ac)
            fold_ts_ac.append(test_ac)
            print('Fold:', i)
            print('Training Accuracy {}: {:.2f}'.format('KNN', training_ac))
            print('Testing Accuracy {}: {:.2f}'.format('KNN', test_ac))

        xrs_validation.write_model_results_to_file('dist_functions_'+ 'KNN' + '_' + str(k), 'tr', fold_tr_ac, [])
        xrs_validation.write_model_results_to_file('dist_functions_'+ 'KNN' + '_' + str(k), 'ts', fold_ts_ac, [])

        avg_ac, se = xrs_validation.get_average_and_se(fold_tr_ac, number_of_folds)
        avg_ac_tfrac['tr'].append(avg_ac)
        se_tfrac['tr'].append(se)
        #print('Average Training Accuracy {}: {:.2f}'.format(get_model_name(model_idx), avg_ac))
        #print('Standard Error {}: {:.2f}'.format(get_model_name(model_idx), se))

        avg_ac, se = xrs_validation.get_average_and_se(fold_ts_ac, number_of_folds)
        avg_ac_tfrac['ts'].append(avg_ac)
        se_tfrac['ts'].append(se)
        print('Average Testing Accuracy {}: {:.2f}'.format('KNN', avg_ac))
        print('Standard Error {}: {:.2f}'.format('KNN', se))

    return avg_ac_tfrac, se_tfrac


def knn(train, test, k=5):
    model = KNN()
    model.fit(train, k)
    return model.get_accuracy(train), model.get_accuracy(test)


if __name__ == "__main__":
    training_data_filename = 'trainingSet.csv'
    test_data_filename = 'testSet.csv'

    if len(sys.argv) > 1:
        training_data_filename = sys.argv[1]
        test_data_filename = sys.argv[2]
        modelIdx = int(sys.argv[3])

    if util.final:
        columns, train = util.readFile(training_data_filename)
        _, test = util.readFile(test_data_filename)
    else:
        columns, train = util.readFile('test_' + training_data_filename)
        _, test = util.readFile('test_' + test_data_filename)

        columns = columns[:5] + [columns[-1]]
        train = train[columns]
        test = test[columns]

    k = [1, 3, 5, 7, 9, 11, 13]
    # model_idxs = [1, 2, 3]
    model_idxs = []

    """
    avg_ac_tfrac, se_tfrac = run_cross_validation_for_k(train, k, 10)
    for x in avg_ac_tfrac:
        xrs_validation.write_model_results_to_file('k_' + 'KNN', x, avg_ac_tfrac[x], se_tfrac[x])

    avg_ac, se = xrs_validation.read_model_results_from_file('k_' + 'KNN', 'ts')
    plt.errorbar(k, avg_ac, yerr=se, label='KNN' + '_' + 'ts' + '_ac', fmt='o-')

    plt.legend(loc='upper right')
    plt.xticks(k)
    plt.tight_layout(pad=3)
    plt.xlabel('k-nearest neighbours used in prediction.')
    plt.ylabel('Model Accuracy')
    plt.savefig('bonus.pdf', format='pdf')
    plt.clf()
    """

    dist_functions = [1, 2, 3]

    # avg_ac_tfrac, se_tfrac = run_cross_validation_for_dist(train, dist_functions, 10)
    # for x in avg_ac_tfrac:
    #     xrs_validation.write_model_results_to_file('dist_functions_' + 'KNN', x, avg_ac_tfrac[x], se_tfrac[x])

    # avg_ac, se = xrs_validation.read_model_results_from_file('dist_functions_' + 'KNN', 'ts')
    # plt.errorbar(k, avg_ac, yerr=se, label='KNN' + '_' + 'ts' + '_ac', fmt='o-')

    # plt.legend(loc='upper right')
    # plt.xticks(dist_functions)
    # plt.tight_layout(pad=3)
    # plt.xlabel('k-nearest neighbours used in prediction.')
    # plt.ylabel('Model Accuracy')
    # plt.savefig('bonus.pdf', format='pdf')
    # plt.clf()

    """
    model = KNN()
    model.fit(train, k=5)
    train_ac = model.get_accuracy(train)
    print('Training Accuracy {}: {:.2f}'.format('KNN', train_ac))
    test_ac = model.get_accuracy(test)
    print('Test Accuracy {}: {:.2f}'.format('KNN', test_ac))
    """

    model = KNN()
    model.fit(train, k=5, dist_function=2)
    train_ac = model.get_accuracy(train)
    print('Training Accuracy {}: {:.2f}'.format('KNN', train_ac))
    test_ac = model.get_accuracy(test)
    print('Test Accuracy {}: {:.2f}'.format('KNN', test_ac))
