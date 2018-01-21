"""
Name: Jared Garcia
Teacher: Burton
Class: CS - 450, 2 - 3pm
Project: Ponder 02
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# KNeighborsClassifier
class KNNClassifier:

    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, data_train, targets_train):

        return KNNModel(data_train, targets_train, self.n_neighbors)


# KNeighborsModel
class KNNModel:

    def __init__(self, data_train, targets_train, k):
        self.data_train = data_train
        self.target_train = targets_train
        self.k = k

    def predict(self, data_test):

        nInputs = np.shape(data_test)[0]
        closest = np.zeros(nInputs)

        k = self.k

        for n in range(nInputs):

            distances = np.sum((self.data_train - data_test[n, :]) ** 2, axis=1)

            indices = np.argsort(distances, axis=0)

            classes = np.unique(self.target_train[indices[:k]])
            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(k):
                    counts[self.target_train[indices[i]]] += 1

                value = np.max(counts)
                length = len(counts)
                for i in range(length):
                    if value == counts[i]:
                        index_value = i - 1
                        closest[n] = classes[index_value]

        return closest


def run_test(data, target, classifier):

    # Split the dataset
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30)

    print("RUNNING TEST SET\n")

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



def get_classifier():

    #classifier = GaussianNB()
    classifier = KNNClassifier(3)
    #classifier = KNeighborsClassifier(n_neighbors=3) # Existing Implementation

    return classifier


def get_dataset():

    iris = datasets.load_iris()
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
    data, target = get_dataset()

    run_test(data, target, classifier)


if __name__ == "__main__":
    main()