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
import random
from sklearn.preprocessing import Imputer
import math


# DecisionTreeClassifier
class NeuralNetClassifier:

    def __init__(self, layers="", number_of_nodes= [], bias_input="", threshold=""):
        self.bias_input = bias_input
        self.number_of_nodes = number_of_nodes
        self.threshold = threshold
        self.layers = layers

    def fit(self, data_train, targets_train, numberOfWeights, numberOfOutputs, classes):

        return NeuralNetModel(self.layers,data_train,targets_train,self.number_of_nodes,numberOfWeights,self.bias_input,self.threshold,numberOfOutputs,classes)


# DecisionTreeModel
class NeuralNetModel:

    def __init__(self, layers="",data_train=[],targets_train=[],numberOfNodes=[],numberOfWeights="",bias_input="",threshold="",numberOfOuputs=0,classes=[]):
        self.layers = layers
        self.classes = classes
        self.numberOfOutputs = numberOfOuputs
        self.data = data_train
        self.target = targets_train
        self.numberOfNodes = numberOfNodes
        self.numberOfWeights = numberOfWeights
        self.bias_input = bias_input
        self.threshold = threshold
        self.nodes = []

    def predict(self, data_test):

        return data_test

    def add_output_layer(self):
        self.numberOfNodes.append(self.numberOfOutputs)

    def create_layer(self):

        self.add_output_layer()

        # Account for bias input
        numberOfWeights = self.numberOfWeights + 1
        print("number of weights", numberOfWeights)

        for k in range(self.layers + 1):
            node_layer = []
            for j in range(self.numberOfNodes[k]):
                weights = []
                if k == 0:
                    # Put in weights for the 1st layer
                    for i in range(numberOfWeights):
                        value = random.uniform(-1, 1)
                        weights.append(value)
                else:
                    # Put in weights for the 2nd layer
                    for i in range(self.numberOfNodes[k - 1] + 1):
                        value = random.uniform(-1, 1)
                        weights.append(value)

                new_node = Node(weights, 0,0,0)
                node_layer.append(new_node)
                print("Layer", k, "node's weight num",len(new_node.weights))
            self.nodes.append(node_layer)
            print("Number of nodes",len(node_layer))

    def iterate_through_network(self):
        rows = np.shape(self.data)[0]
        print("Rows:", rows)

        cols = self.data.shape[1]
        print("Cols:", cols)

        predictions = []
        for i in range(0, rows):
            inputs = []
            inputs.append(self.bias_input)
            for j in range(0, cols):
                inputs.append(self.data[i][j])

            maxActivation = 0;
            predicted_node = 0;
            for k in range(self.layers + 1):
                for t in range(len(self.nodes[k])):
                    if k == 0:
                        print("Weights",self.nodes[k][t].weights)
                        print("Inputs", inputs)
                        multiplied_list = map(lambda x, y: x * y, self.nodes[k][t].weights, inputs)
                        output = sum(multiplied_list)
                        print("h",output)
                        activation = (1 / (1 + math.exp(- (output))))
                        self.nodes[k][t].activation = activation
                        print("a",activation)

                    else:
                        activation_inputs = []
                        activation_inputs.append(self.bias_input)
                        for n in range(len(self.nodes[k - 1])):
                            activation_inputs.append(self.nodes[k - 1][n].activation)

                        print("Weights", self.nodes[k][t].weights)
                        print("Activation", activation_inputs)
                        multiplied_list = map(lambda x, y: x * y, self.nodes[k][t].weights, activation_inputs)
                        output = sum(multiplied_list)
                        print("h",output)
                        activation = (1 / (1 + math.exp(- (output))))
                        self.nodes[k][t].activation = activation
                        print("a",activation)

                        if self.layers == k:
                            if maxActivation == 0:
                                maxActivation = activation
                                predicted_node = t
                            elif maxActivation < activation:
                                maxActivation = activation
                                predicted_node = t

            print("Predicted",self.classes[predicted_node],"Highest Activation",maxActivation,"\n")
            predictions.append(self.classes[predicted_node])

        return predictions


# Individual Node
class Node:

    def __init__(self, weights=[],activation=0,value=0,error=0):
        self.weights = weights
        self.value = value
        self.activation = activation
        self.error = error


def run_test(data, target, classifier):

    cols = data.shape[1]
    print(cols)

    unique = list(set(target))
    print(unique)
    number_of_outputs = len(unique)
    print(number_of_outputs)

    model = classifier.fit(data, target, cols, number_of_outputs, unique)

    model.create_layer()

    weights = [-.2, .5, -.3]
    inputs = [-1, .4, .2]
    multiplied_list = map(lambda x, y: x * y, weights, inputs)
    output = sum(multiplied_list)
    print(output)

    activation = (1 / (1 + math.exp(- (output))))

    print(activation)

    targets_predicted = model.iterate_through_network()

    print("Target Predicted\n", targets_predicted)
    print("Target Test\n", target, "\n")

    # Correct/percentage
    correct = (targets_predicted == target).sum()
    length = len(target)
    percentage = (round(((correct / length) * 100), 2))
    print("Results")
    print("Correct:", correct)
    print("Out of:", correct, "/", length)
    print("Percentage:", percentage, "% accurate\n")


    # Send some data into predict from the tree

    # Split the dataset
    # data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30)
    #
    # print("RUNNING TEST SET\n")
    #
    # model = classifier.fit(data_train, target_train)
    #
    # Find Target Data
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


def k_fold_cross_validation(classifier, data, target):

    scores = cross_val_score(classifier, data, target, cv=20, scoring='f1_macro')
    print("Individual Scores", np.round(scores * 100, 2))

    print("Average Score", round(np.average(scores) * 100, 2),"%")


def get_classifier():
    # inputs are determined by columns
    # outputs are determined by classes
    layers = 2
    number_of_nodes = []
    number_of_nodes.append(2)
    number_of_nodes.append(3)
    bias_input = -1
    threshold = 0

    classifier = NeuralNetClassifier(layers, number_of_nodes, bias_input, threshold)

    return classifier


def normalize_data(data):

    z_scores = (data - data.mean()) / data.std()

    return z_scores


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
    data = normalize_data(data)

    return data, target


def get_iris_dataset():

    iris = datasets.load_iris()

    data = normalize_data(iris.data)

    # print("iris",iris)
    #
    # # Show the data (the attributes of each instance)
    #print("All of the data\n",iris.data, "\n")
    #print("Z_scores data",data)
    #print("Targets",iris.target)
    #
    # # Show the target values (in numeric format) of each instance
    # print("Target Data\n",iris.target, "\n")
    #
    # # Show the actual target names that correspond to each number
    # print("Target Names\n",iris.target_names, "\n")

    return data, iris.target


def get_diabetes_dataset():

    df = pd.read_csv("diabetes.csv", header=None)

    # print((df[[1, 2, 3, 4, 5]] == 0).sum())
    df[[1, 2, 3, 4, 5]] = df[[1, 2, 3, 4, 5]].replace(0, np.NaN)
    # print((df[[1, 2, 3, 4, 5]] == 0).sum())

    data = df.values

    imputer = Imputer(strategy='mean', axis=0)
    data = imputer.fit_transform(data)

    target = data[:, -1:]
    target = target.flatten()

    data = data[:, :8]

    print(data)
    # print(target)

    data = normalize_data(data)
    #print(data)
    #print(target)

    return data, target


def main():

    classifier = get_classifier()
    #data, target = get_iris_dataset()
    data, target = get_diabetes_dataset()
    #data, target = get_loan_data()
    run_test(data, target, classifier)


if __name__ == "__main__":
    main()