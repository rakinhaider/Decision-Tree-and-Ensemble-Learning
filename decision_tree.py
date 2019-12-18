import pandas as pd
import numpy as np
import json
import utils as util


class DecisionTreeNode:
    def __init__(self, node_id, attribute=None, values=None, children=None, label=None):
        if values is None:
            values = []
        if children is None:
            children = []
        if label is None:
            self.node_id = node_id
            self.attribute = attribute
            self.values = values
            self.is_leaf = False
            self.children = children
        else:
            self.node_id = node_id
            self.label = label
            self.is_leaf = True
            self.children = []
            self.values = []

    def __str__(self):
        if not self.is_leaf:
            d = {
                'node_id': self.node_id,
                'is_leaf': self.is_leaf,
                'attribute': self.attribute,
                'values': [str(v) for v in self.values],
                'children': [str(ch) for ch in self.children]
            }
            return json.dumps(d)
        else:
            d = {'label': str(self.label),
                 'node_id:': str(self.node_id)}
            return json.dumps(d)

    def add_child_for_value(self, value, child):
        if not self.is_leaf:
            self.values.append(value)
            self.children.append(child)

    def add_child(self, child):
        if not self.is_leaf:
            self.children.append(child)

    def add_children(self, children):
        if not self.is_leaf:
            self.children.append(children)

    def predict(self, data):
        if not self.is_leaf:
            if data[self.attribute] in self.values:
                value_idx = self.values.index(data[self.attribute])
                ch = self.children[value_idx]
                pred = ch.predict(data)
            else:
                pred = None
            return pred
        else:
            return self.label


class DecisionTree:
    def __init__(self):
        self.root = None
        self.nodes = 0
        self.majority_label = None

    def __str__(self):
        return str(self.root)

    def fit(self, data, depth=8):
        attributes = list(data.columns)[:-1]
        sample_limit = 0 if not util.final else 50
        self.majority_label = data[data.columns[-1]].value_counts(sort=True).index[0]
        # print(self.majority_label)
        self.root = self.build_tree(data, attributes, depth_limit=depth, sample_limit=sample_limit)

    def get_gini(self, examples):
        target_attribute = examples.columns[-1]
        counts = examples[target_attribute].value_counts()
        total = len(examples)
        gini = 1
        for val in counts.index:
            probab = counts[val] * 1.0 / total
            gini = gini - probab * probab

        return gini

    def get_gini_gain(self, s, attr):
        gini_s = self.get_gini(s)
        values = s[attr].unique()
        gain = gini_s
        for val in values:
            s_v = s[s[attr] == val]
            gain = gain - len(s_v) * self.get_gini(s_v) * 1.0 / len(s)

        return gain

    def get_best_attribute(self, examples, attributes, criteria):
        if criteria == 1:
            gains = []
            for attr in attributes:
                gains.append(self.get_gini_gain(examples, attr))

            return attributes[np.argmax(gains)]

    def build_tree(self, examples, attributes, depth_limit=-1, sample_limit=-1):
        target_attribute = list(examples.columns)[-1]
        # print(target_attribute)
        # print(examples)
        # print(len(examples[target_attribute].unique()))
        # print(examples[target_attribute].unique()[0])
        # print(examples[target_attribute].value_counts(sort=True).index[0])

        if len(examples[target_attribute].unique()) == 1:
            unique_label = examples[target_attribute].unique()[0]
            self.nodes = self.nodes + 1
            dcn = DecisionTreeNode(label=unique_label, node_id=self.nodes)
            # print('Unique label:', dcn)
            return dcn
        elif len(attributes) == 0 or depth_limit == 0 or len(examples) <= sample_limit:
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
        attributes.remove(best_attribute)
        rem_attributes = [i for i in attributes]
        values = np.sort(examples[best_attribute].unique())
        self.nodes = self.nodes + 1
        dcn = DecisionTreeNode(attribute=best_attribute, node_id=self.nodes)
        for val in values:
            s_v = examples[examples[best_attribute] == val]
            print(best_attribute, val, len(rem_attributes), depth_limit)
            child_node = self.build_tree(s_v, rem_attributes, depth_limit - 1, sample_limit)
            # print('child:', child_node)
            dcn.add_child_for_value(val, child_node)
            # print('parent:', dcn)

        return dcn

    def predict(self, test):
        predictions = []
        for _, row in test.iterrows():
            pred = self.root.predict(row)
            if pred is not None:
                predictions.append(pred)
            else:
                predictions.append(self.majority_label)

        return predictions

    def get_accuracy(self, test):
        predictions = self.predict(test)
        target_attribute = test.columns[-1]
        correct = (test[target_attribute] == predictions).astype(int).value_counts()[1]
        return correct * 100 / len(test)
