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
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer

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
                counts = np.zeros(len(classes))
                #print("classes", classes)
                #print("counts", counts)
                for i in range(k):
                    item = self.target_train[indices[i]]
                    #print(item)
                    class_length = len(classes)
                    for t in range(class_length):
                        if item == classes[t]:
                            counts[t] += 1

                #print("after", counts)
                value = np.max(counts)
                length = len(counts)
                for i in range(length):
                    if value == counts[i]:
                        index_value = i
                        closest[n] = classes[index_value]
                        #print("item",classes[index_value])

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
    percentage = (round(((correct / length) * 100), 2))
    print("Results")
    print("Correct:",correct)
    print("Out of:",correct,"/",length)
    print("Percentage:",percentage,"% accurate\n")

    # Only use on existing classifiers from the provided libraries, will not work on self-built classifiers
    k_fold_cross_validation(classifier, data, target)


def k_fold_cross_validation(classifier, data, target):

    scores = cross_val_score(classifier, data, target, cv=20, scoring='f1_macro')
    print("Individual Scores", np.round(scores * 100, 2))

    print("Average Score", round(np.average(scores) * 100, 2),"%")

def get_classifier():

    # classifier = GaussianNB() # Existing Implementation
    #classifier = KNNClassifier(8)
    classifier = KNeighborsClassifier(n_neighbors=8) # Existing Implementation

    return classifier


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


def get_car_dataset():

    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "condition"]

    df = pd.read_csv("car.csv", header=None, names=headers, index_col=False)

    df.head()

    obj_df = df.select_dtypes(include=['object']).copy()

    cleanup_nums = {
        "buying": {"vhigh": 1, "high": 2, "med": 3, "low": 4},
        "maint": {"vhigh": 1, "high": 2, "med": 3, "low": 4},
        "doors": {'2': 2, '3': 3, '4': 4, "5more": 5},
        "persons": {"2": 2, "4": 4, "more": 5},
        "lug_boot": {"small": 1, "med": 2, "big": 3},
        "safety": {"low": 1, "med": 2, "high": 3},
        "condition": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
        }

    obj_df.replace(cleanup_nums, inplace=True)

    data = obj_df.values
    target = data[:, -1:]
    target = target.flatten()

    data = data[:, :6]

    return data, target


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

    # print(data)
    # print(target)

    return data, target


def get_mpg_dataset():
    # 1.   mpg: continuous
    #  2.  cylinders: multi - valued  discrete
    # 3.  displacement: continuous
    # 4.  horsepower: continuous
    # 5.  weight: continuous
    # 6. acceleration: continuous
    # 7. model year: multi - valued  discrete
    # 8.  origin: multi - valued   discrete
    # 9.  car  name: string(unique for each instance)

    # headers = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model", "origin", "name"]

    df = pd.read_csv("mpg.csv", header=None, index_col=False, na_values="?")

    df.head()
    df.dropna(inplace=True, axis=1)


    # obj_df = df.select_dtypes(include=['object']).copy()
    # obj_df["name"] = obj_df["name"].astype('category')
    #
    # print(obj_df)
    # obj_df["name_cat"] = obj_df["name"].cat.codes

    print(df)
    data = df.values
    rows = np.shape(data)[0]
    data = np.reshape(data, (rows, 9))
    print(data)

    target = data[:, -1:]
    target = target.flatten()

    data = data[:, :8]

    # print(data)
    # print(target)

    return data, target


def main():

    classifier = get_classifier()
    #data, target = get_iris_dataset()
    #data, target = get_car_dataset()
    data, target = get_diabetes_dataset()
    #data, target = get_mpg_dataset()   # Function not usable, there are a few errors.
    run_test(data, target, classifier)


if __name__ == "__main__":
    main()