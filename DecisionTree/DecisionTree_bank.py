import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter

column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
df1 = pd.read_csv("datasets/bank/train.csv", names=column_headers)
X_train = df1.drop('label', axis=1)
y_train = df1['label']

df2 = pd.read_csv("datasets/bank/test.csv", names=column_headers)
X_test = df2.drop('label', axis=1)
y_test = df2['label']

# Add a threshold dictionary for numerical attributes
thresholds = {'age': np.median(X_train['age']), 'balance': np.median(X_train['balance']),
              'day': np.median(X_train['day']), 'duration': np.median(X_train['duration']),
              'campaign': np.median(X_train['campaign']), 'pdays': np.median(X_train['pdays']),
              'previous': np.median(X_train['previous'])}

class TreeNode:
    def __init__(self, attribute, attributeName, is_leaf, label, depth, info_gain, entropy_parent_attr, parent_attr_val):
        self.attribute = attribute
        self.attributeName = attributeName
        self.children = {}
        self.is_leaf = is_leaf
        self.label = label
        self.depth = depth
        self.info_gain = info_gain
        self.entropy_parent_attr = entropy_parent_attr
        self.parent_attr_val = parent_attr_val

    def get_attribute(self):
        return self.attribute

    def add_child(self, child_node, attr_value):
        self.children[attr_value] = child_node

    def predict(self, x):
        if self.is_leaf:
            return self.label

        current_val = x.iloc[self.attribute]  # Use .iloc to access DataFrame columns by index

        if self._is_numerical(self.attribute):
            current_val = self._compare_threshold(x, self.attribute)

        if current_val not in self.children:
            return self.label

        return self.children[current_val].predict(x)

    def print_node(self, space=""):
        print(f"{space}Depth: {self.depth}")
        print(f"{space}Selected Feature: {self.attributeName}")
        print(f"{space}Information Gain for Parent Feature: {self.info_gain}")
        print(f"{space}Entropy for Parent Feature: {self.entropy_parent_attr}")
        print(f"{space}Parent Feature Value: {self.parent_attr_val}")
        print(f"{space}Label: {self.label}")
        for child in self.children.values():
            child.print_node(space + "\t")

    def _is_numerical(self, attribute):
        return attribute in thresholds

    def _compare_threshold(self, x, attribute):
        if x.iloc[attribute] >= thresholds[attribute]:
            return ">= " + str(thresholds[attribute])
        else:
            return "< " + str(thresholds[attribute])

    def _majority_error(self, X, y, attribute):
        values = set(X[attribute])
        return sum([(X[attribute] == value).mean() *
            (1 - Counter(y[X[attribute] == value]).most_common(1)[0][1] / len(y[X[attribute] == value]))
                    for value in values])

    def _gini(self, X, y, attribute):
        values = set(X[attribute])
        gini = 1
        for value in values:
            p = (X[attribute] == value).mean()
            gini -= p**2
        return gini

class DecisionTreeClassifier:
    def __init__(self, max_depth=np.inf):
        self.root = None
        self.depth = 0
        if max_depth < 1:
            print("max_depth cannot be lower than 1! Setting it to 1.")
            max_depth = 1
        self.max_depth = max_depth
        self.longest_path_len = 0

    def build_tree(self, X, Y, attribute_names, attribute_list=[], current_depth=0,
                   parent_info={"max_info_gain": None, "attribute_list[max_attribute]": None, "value": None}):
        if current_depth > self.longest_path_len:
            self.longest_path_len = current_depth
        if current_depth >= self.max_depth or len(attribute_list) == 0 or len(np.unique(Y)) == 1:
            vals, counts = np.unique(Y, return_counts=True)
            return TreeNode(None, None, True, vals[np.argmax(counts)], current_depth,
                            parent_info["max_info_gain"], parent_info["attribute_list[max_attribute]"],
                            parent_info["value"])

        max_info_gain = -1
        max_attribute = None
        i = 0
        for attribute in attribute_list:
            info_gain, entropy_attribute, entropy_parent = self.calculate_information_gain(X, Y, attribute)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_attribute = i
                entropy = entropy_parent
            i += 1

        vals, counts = np.unique(Y, return_counts=True)
        root = TreeNode(attribute_list[max_attribute], attribute_names[attribute_list[max_attribute]],
                        False, vals[np.argmax(counts)], current_depth,
                        parent_info["max_info_gain"], parent_info["attribute_list[max_attribute]"],
                        parent_info["value"])

        attribute_values = np.unique(X.iloc[:, attribute_list[max_attribute]])  # Use .iloc to access DataFrame columns by index
        new_attribute_list = np.delete(attribute_list, max_attribute)
        for value in attribute_values:
            indices = np.where(X.iloc[:, attribute_list[max_attribute]] == value)[0]
            if len(indices) == 0:
                root.add_child(TreeNode(None, None, True, vals[np.argmax(counts)], current_depth + 1,
                                        max_info_gain, attribute_list[max_attribute], value), current_depth)
            else:
                parent_info = {
                    "max_info_gain": max_info_gain,
                    "attribute_list[max_attribute]": entropy,
                    "value": value
                }
                root.add_child(self.build_tree(X.iloc[indices], Y.iloc[indices], attribute_names, new_attribute_list,
                                               current_depth + 1, parent_info), value)
        return root

    def calculate_entropy(self, counts):
        total = sum(counts)
        entropy_value = 0
        for element in counts:
            p = (element / total)
            if p != 0:
                entropy_value -= p * np.log2(p)
        return entropy_value

    def calculate_information_gain(self, X, Y, attribute):
        _, counts = np.unique(Y, return_counts=True)
        entropy_attribute = self.calculate_entropy(counts)
        entropy_parent = 0
        distinct_attr_values = list(set(X.iloc[:, attribute]))  # Use .iloc to access DataFrame columns by index
        for val in distinct_attr_values:
            indices = np.where(X.iloc[:, attribute] == val)[0]
            _, counts = np.unique(Y.iloc[indices], return_counts=True)
            entr = self.calculate_entropy(counts)
            entropy_parent += (len(indices) / len(Y)) * entr
        info_gain = entropy_attribute - entropy_parent
        return info_gain, entropy_attribute, entropy_parent

    def fit(self, X, Y):
        attribute_names = list(range(X.shape[1]) if isinstance(X, pd.DataFrame) else range(X.shape[1]))  # Assume attributes are indexed
        attribute_list = np.arange(X.shape[1])
        self.root = self.build_tree(X, Y, attribute_names, attribute_list, 0)

    def predict(self, X):
        predictions = []
        for _, x in X.iterrows():
            predictions.append(self.root.predict(x))
        return predictions

    def get_longest_path_len(self):
        return self.longest_path_len

    def get_root_attribute(self):
        if self.root:
            return self.root.get_attribute()
        return None

    def print_tree(self):
        self.root.print_node("")

results = {'information_gain': [], 'majority_error': [], 'gini': []}

for max_depth in range(1, 12):
    for criterion in ['information_gain', 'majority_error', 'gini']:
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        results[criterion].append((1 - train_acc, 1 - test_acc))

# Print results
print('Criterion      Train   Test')
for criterion, errors in results.items():
    print(f'{criterion: <15} {errors[-1][0]:.3f} {errors[-1][1]:.3f}')
