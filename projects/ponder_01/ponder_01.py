"""
Name: Jared Garcia
Teacher: Burton
Class: CS - 450, 2 - 3pm
Project: Ponder 01
"""

import numpy as np
import csv
import re
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def iris_machine_learning_algo():
    iris = datasets.load_iris()

    # Show the data (the attributes of each instance)
    print(iris.data,"\n")

    # Show the target values (in numeric format) of each instance
    print(iris.target,"\n")

    # Show the actual target names that correspond to each number
    print(iris.target_names,"\n")

    # Split the dataset
    data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.30)
    """
    print("Data Train\n",data_train,"\n")
    print("Data Test\n",data_test,"\n")
    print("Target Train\n",target_train,"\n")
    print("Target Test\n",target_test,"\n")
    """

    # Train the dataset and create a model
    classifier = GaussianNB()
    model = classifier.fit(data_train, target_train)

    # Find Target Data
    targets_predicted = model.predict(data_test)
    print("Target Predicted\n",targets_predicted)
    print("Target Test\n",target_test,"\n")

    # Correct/percentage
    correct = (targets_predicted == target_test).sum()
    length = len(target_test)
    percentage = ("%.2f" % round(((correct / length) * 100), 2))
    print("Results")
    print("Correct:",correct)
    print("Out of:",correct,"/",length)
    print("Percentage:",percentage,"% accurate")

    return data_train, data_test, target_test, target_train


# HardCodedClassifier
class HardCodedClassifier:

    def __init__(self):
        pass

    def fit(self, data_train, targets_train):
        return HardCodedModel()


# HardCodedModel
class HardCodedModel:

    def __init__(self):
        pass

    def predict(self, data_test):
        targets_predicted = []
        for i in data_test:
            targets_predicted.append(0)

        return targets_predicted


def hardCodedAlgo(data_train, target_train, data_test, target_test):
    # Test my classes
    print()
    print()
    print("Testing Hardcoded Classes")
    classifier = HardCodedClassifier()
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)

    print("Hardcoded Prediction",targets_predicted)
    print("Target Test\n",target_test,"\n")

    # Correct/percentage
    correct = (targets_predicted == target_test).sum()
    length = len(target_test)
    percentage = ("%.2f" % round(((correct / length) * 100), 2))
    print("Results")
    print("Correct:", correct)
    print("Out of:", correct, "/", length)
    print("Percentage:", percentage, "% accurate\n\n")


def stretch():
    # Pulls from text file using numpy
    print("Stretch Reading from text file")
    data = np.genfromtxt("projects/ponder_01/test.txt", delimiter=" ", dtype="f")
    # result = re.sub('[^0-9]', '', data)
    print(data)

    # Pulls from text file using open
    with open('projects/ponder_01/test.txt', 'r') as file:
        lines = file.read().split(' ')
        # non_decimal = re.compile(r'[^\d.]+')
        print(lines)


def main():
    data_train, data_test, target_test, target_train = iris_machine_learning_algo()
    hardCodedAlgo(data_train, target_train, data_test, target_test)
    stretch()


if __name__ == "__main__":
    main()