"""
Name: Jared Garcia
Teacher: Burton
Class: CS - 450, 2 - 3pm
Project: Ponder 04
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
import math


# DecisionTreeClassifier
class DecisionTreeClassifier:

    def __init__(self):
        pass

    def fit(self, data_train, targets_train, headers):

        return DecisionTreeModel(data_train, targets_train, headers)


# DecisionTreeModel
class DecisionTreeModel:

    def __init__(self, data_train, targets_train, headers):
        self.data = data_train
        self.target = targets_train
        self.headers = headers
        self.root = None

    def build_tree(self, node):

        print("\nRecursive")

        if node == None:
            root = self.find_root()
            self.build_tree(root)
        elif len(node.target_values) == 0:
            print("Died")
            return
        else:
            print("Start", node.attribute, node.target_values, node.branches, node.child_nodes)
            for u in range(len(node.branches)):
                print("\nindex",u)
                node_name = []
                entropies = []
                target_values = []
                node_branches = []
                children_nodes = []
                node_attributes = []

                for i in range(len(node.target_values)):
                    variety = []
                    for j in range(len(node.branches[u])):
                        variety.append(self.data[node.branches[u][j]][node.target_values[i]])

                    unique = list(set(variety))
                    total_entropy = 0
                    branches_with_rows = []
                    get_children = []
                    attributes = []
                    print('variety', variety,"branch",node.branches[u])
                    for x in range(len(unique)):
                        print("go")
                        classes = []
                        row_values = []
                        for y in range(len(variety)):
                            if unique[x] == variety[y]:
                                classes.append(self.target[node.branches[u][y]])
                                row_values.append(node.branches[u][y])

                        print('rows', row_values)
                        entropy = calculate_entropy(classes, len(node.branches[u]))
                        if entropy == 0:
                            child_node = Node("", max(classes), [], [], [])
                            print("Created child",child_node, "Max classes",max(classes))
                            get_children.append(child_node)
                            branches_with_rows.append([])
                            attributes.append([])
                        else:
                            branches_with_rows.append(row_values)
                            attributes.append(unique[x])
                        total_entropy += entropy
                    print("Total Entropy ", total_entropy,"\n")

                    node_name.append(self.headers[node.target_values[i]])
                    entropies.append(total_entropy)
                    node_branches.append(branches_with_rows)
                    children_nodes.append(get_children)
                    node_attributes.append(attributes)

                val = np.min(entropies)
                n_name = ""
                className = ""
                branches = []
                children = []
                attrs = []

                for e in range(len(entropies)):
                    if val == entropies[e]:
                        n_name = node_name[e]
                        branches = node_branches[e]
                        children = children_nodes[e]
                        attrs = node_attributes[e]
                    else:
                        target_values.append(node.target_values[e])

                if len(branches) != 0:
                    print("New Node",n_name, val, target_values, branches, children, attrs)
                    new_node = Node(n_name, className, target_values, branches, children,attrs)
                    node.child_nodes.append(new_node)
                    node.branches[u] = []
                    print(node.attribute, node.target_values, node.branches, node.child_nodes, node.attributeValue)

                    self.build_tree(new_node)

    def predict(self, data_test):

        return data_test

    def find_root(self):
        rows = np.shape(self.data)[0]
        print("Rows:", rows)

        cols = self.data.shape[1]
        print("Cols:", cols)
        node_name = []
        entropies = []
        target_values = []
        node_branches = []
        node_attributes = []

        for i in range(0, cols):
            variety = []
            for j in range(0, rows):
                variety.append(self.data[j][i])

            unique = list(set(variety))
            entropy = 0
            branches_with_rows = []
            attributes = []
            for x in range(len(unique)):
                classes = []
                row_values = []
                for y in range(len(variety)):
                    if unique[x] == variety[y]:
                        classes.append(self.target[y])
                        row_values.append(y)

                entropy += calculate_entropy(classes, rows)
                branches_with_rows.append(row_values)
                attributes.append(unique[x])

            print("Total Entropy ", entropy, "\n")
            node_name.append(self.headers[i])
            entropies.append(entropy)
            node_branches.append(branches_with_rows)
            node_attributes.append(attributes)

        val = np.min(entropies)
        root_name = ""
        branches = []
        attrs = []

        for e in range(len(entropies)):
            if val == entropies[e]:
                root_name = node_name[e]
                branches = node_branches[e]
                attrs = node_attributes[e]
            else:
                target_values.append(e)

        print(root_name, val, target_values, branches, attrs)
        self.root = Node(root_name, "",target_values, branches, [], attrs)
        node = self.root

        return node


# Individual Node
class Node:

    def __init__(self, attribute="", className="", target_values=[], branches=[], child_nodes=[], attributeValue=[]):
        self.attribute = attribute  # Attribute Name, ex. Income,
        self.className = className # Final leaf Name, ex. Yes or No
        self.branches = branches      # Branch, all rows for that attribute column
        self.target_values = target_values   # Target Values, the columns to evaluate Income, Collateral
        self.child_nodes = child_nodes       # Nodes, connected to the this node
        self.attributeValue = attributeValue # Incomes attributes, High or Low


def run_test(data, target, headers, classifier):

    model = classifier.fit(data, target, headers)

    model.build_tree(None)


    # Send some data into predict from the tree

    # Split the dataset
    # data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30)
    #
    # print("RUNNING TEST SET\n")
    #
    # model = classifier.fit(data_train, target_train)
    #
    # # Find Target Data
    # targets_predicted = model.predict(data_test)
    # print("Target Predicted\n",targets_predicted)
    # print("Target Test\n",target_test,"\n")
    #
    # # Correct/percentage
    # correct = (targets_predicted == target_test).sum()
    # length = len(target_test)
    # percentage = (round(((correct / length) * 100), 2))
    # print("Results")
    # print("Correct:",correct)
    # print("Out of:",correct,"/",length)
    # print("Percentage:",percentage,"% accurate\n")

    # Only use on existing classifiers from the provided libraries, will not work on self-built classifiers
    #k_fold_cross_validation(classifier, data, target)


def find_class(test_date, root):

    if root.className != "":
        return root.className

    for i in range(test_date):
        for e in range(len(root.attributeValue)):
            if root.attributeValue[e] == test_date[i]:
                return 1


def k_fold_cross_validation(classifier, data, target):

    scores = cross_val_score(classifier, data, target, cv=20, scoring='f1_macro')
    print("Individual Scores", np.round(scores * 100, 2))

    print("Average Score", round(np.average(scores) * 100, 2),"%")


# function to associate the target values with the classes and
#     separate that data with those classes


def calculate_entropy(data, rows):

    no_duplicates = list(set(data))
    print("Classes",data)
    print("Proportion",len(data),rows)
    print("Ratio",len(data)/rows)
    totals = []

    for index1 in range(len(no_duplicates)):
        total = 0
        for index2 in range(len(data)):
            if (no_duplicates[index1] == data[index2]):
                total += 1
        totals.append(total)

    entropy = 0
    total_value = sum(totals)
    for index in range(len(totals)):
        value = totals[index]
        proportion = value/total_value
        entropy += -(proportion) * math.log2(proportion)

    print("Individual Entropy",entropy)
    entropy = entropy * (len(data)/rows)

    return entropy


def get_classifier():

    classifier = DecisionTreeClassifier()

    return classifier


def get_loan_data():

    headers = ["Credit Score", "Income", "Collateral", "Loan"]

    df = pd.read_csv("loan.csv", header=None, names=headers, index_col=False)

    df.head()

    print("First\n",df)

    obj_df = df.select_dtypes(include=['object']).copy()

    cleanup_nums = {
        "Credit Score": {"Good": 2, "Average": 3, "Low": 4},
        "Income": {"High": 5, "Low": 6},
        "Collateral": {"Good": 7, "Poor": 8},
        "Loan": {"No": 0, "Yes": 1},
        }

    obj_df.replace(cleanup_nums, inplace=True)

    print("After\n",obj_df)

    data = obj_df.values
    target = data[:, -1:]
    target = target.flatten()

    print("Target\n",target)
    data = data[:, :3]
    print("Data\n", data)

    return data, target, headers


def get_iris_dataset():

    iris = datasets.load_iris()

    # print("iris",iris)
    #
    # # Show the data (the attributes of each instance)
    # print("All of the data\n",iris.data, "\n")
    #
    # # Show the target values (in numeric format) of each instance
    # print("Target Data\n",iris.target, "\n")
    #
    # # Show the actual target names that correspond to each number
    # print("Target Names\n",iris.target_names, "\n")

    return iris.data, iris.target


def main():

    classifier = get_classifier()
    #data, target = get_iris_dataset()
    data, target, headers = get_loan_data()
    run_test(data, target, headers, classifier)


if __name__ == "__main__":
    main()