import pandas as pd
import numpy as np
from scipy import stats
from decision_tree import DecisionTree, DecisionTreeNode
import random
import math


class Bagging:
    def __init__(self, base_classifier):
        self.models = []
        self.base_classifier = base_classifier

    def fit(self, data, m=30, depth=8):
        for i in range(m):
            print('Model #:', i)
            data_m = data.sample(n=len(data), replace=True, random_state=47 + i)
            #data_m = data.sample(n=len(data), replace=True)
            model = self.base_classifier()
            model.fit(data_m, depth=depth)
            self.models.append(model)

    def predict(self, data):
        base_predictions = []
        for model in self.models:
            predictions = model.predict(data)
            base_predictions.append(predictions)

        mode, count = stats.mode(base_predictions)
        return mode[0]

    def get_accuracy(self, data):
        predictions = self.predict(data)
        target_attribute = data.columns[-1]
        correct = (data[target_attribute] == predictions).astype(int).value_counts()[1]
        return correct * 100 / len(data)


class RandomDecisionTree(DecisionTree):

    def __init__(self):
        super().__init__()
        self.path = []

    def get_rem_attributes(self, attributes, remove_attr):
        p = len(attributes)
        attr_samp = random.sample(list(attributes), k=int(math.sqrt(p)))
        # print('attributes:', attributes)
        # print('attr_samp:', attr_samp)
        # print('remove_attr:', remove_attr)
        for attr in remove_attr:
            if attr in attr_samp:
                attr_samp.remove(attr)
        return attr_samp

    def build_tree(self, examples, attributes, depth_limit=-1, sample_limit=-1):
        target_attribute = list(examples.columns)[-1]

        if len(examples[target_attribute].unique()) == 1:
            unique_label = examples[target_attribute].unique()[0]
            self.nodes = self.nodes + 1
            dcn = DecisionTreeNode(label=unique_label, node_id=self.nodes)
            # print('Unique label:', dcn)
            return dcn
        elif len(attributes) == 0 or depth_limit == 0 or len(examples) < sample_limit:
            majority_label = examples[target_attribute].value_counts(sort=True).index[0]
            self.nodes = self.nodes + 1
            dcn = DecisionTreeNode(label=majority_label, node_id=self.nodes)
            """
            if len(attributes) == 0:
                print('No attribute:', dcn)
            elif depth_limit == 0:
                print('No depth:', dcn)
            elif len(examples) < sample_limit:
                print('Low samples:', dcn)
    
            """
            return dcn

        assert (depth_limit > 0)
        assert (len(examples) >= sample_limit)

        best_attribute = self.get_best_attribute(examples, attributes, criteria=1)
        self.path = self.path + [best_attribute]
        rem_attributes = self.get_rem_attributes(examples.columns[:-1], self.path)
        values = np.sort(examples[best_attribute].unique())
        self.nodes = self.nodes + 1
        dcn = DecisionTreeNode(attribute=best_attribute, node_id=self.nodes)
        for val in values:
            s_v = examples[examples[best_attribute] == val]
            child_node = self.build_tree(s_v, rem_attributes, depth_limit - 1, sample_limit)
            # print('child:', child_node)
            dcn.add_child_for_value(val, child_node)
            # print('parent:', dcn)

        self.path.remove(best_attribute)
        return dcn


class RandomForests(Bagging):

    def __init__(self):
        self.models = []
        self.base_classifier = RandomDecisionTree
        # print(self.models)
        # print(self.__dict__)