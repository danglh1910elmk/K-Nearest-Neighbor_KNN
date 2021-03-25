import numpy as np
from sklearn import datasets
import operator

def hightest_votes(k_labels_list): # return a number [1,0,0,0,1,2,2,1,1,2,0]
    # 1st way
    # reduce_array = list(set(k_labels_list))
    # amount_each_element = []
    # for i in range(len(reduce_array)):
    #     amount_each_element.append(np.count_nonzero(k_labels_list == reduce_array[i]))
    # x = reduce_array[amount_each_element.index(max(amount_each_element))]
    # # print('k_labels_list = ',k_labels_list,', choose : ', x)
    # return x

    # 2nd way
    labels_count = [0,0,0]
    for label in k_labels_list:
        labels_count[label] += 1
    
    max_count = max(labels_count)
    print(k_labels_list, '-->', labels_count.index(max_count))
    return labels_count.index(max_count)

def get_k_neighbors(training_X, label_y, point, k): # return a list
    # 1st way:
    # distances = []
    # label_y_copy = label_y.copy() # note
    # n = training_X.shape[0]
    # for p in training_X:
    #     distances.append(np.linalg.norm(point - p, 2))
    
    # for i in range(n-1):
    #     for j in range(i,n):
    #         if distances[i]>distances[j]:
    #             t = distances[i]
    #             distances[i] = distances[j]
    #             distances[j] = t

    #             t = label_y_copy[i]
    #             label_y_copy[i] = label_y_copy[j]
    #             label_y_copy[j] = t
    # return label_y_copy[:k]

    # 2nd way: anh Dung
    # distances = []
    # neighbors = []
    # for i in range(len(training_X)):
    #     distances.append((np.linalg.norm(point - training_X[i], 2), label_y[i]))  # (distance,label)
    # # use external library
    # distances.sort(key=operator.itemgetter(0)) # sort by distance
    # for i in range(k):
    #     neighbors.append(distances[i][1])
    # return neighbors
    
    # 3th way
    # distances = []
    # for p in training_X:
    #     distances.append(np.linalg.norm(point - p, 2))
    # index = []
    # neighbors = []
    # while len(neighbors) < k:
    #     min_distance = 9999
    #     min_index = 0
    #     for i in range(len(distances)):
    #         if i in index:
    #             continue
    #         if distances[i] < min_distance:
    #             min_distance = distances[i]
    #             min_index = i
    #     index.append(min_index)
    #     neighbors.append(label_y[min_index])
    # return neighbors

    # 4th way : shortest way
    distances = []
    label_y_copy = label_y.copy()
    for p in training_X:
        distances.append(np.linalg.norm(point - p, 2))
    distances, label_y_copy = zip(*sorted(zip(distances, label_y_copy)))
    return label_y_copy[:k]

def predict(training_X, label_y, point, k):  # return label of a point
    neighbors_labels = get_k_neighbors(training_X, label_y, point, k)
    return hightest_votes(neighbors_labels)

def accuracy_score(y_predict, groundTruth): # accuracy of the algorithm
    count = 0
    for i in range(len(y_predict)):
        if y_predict[i] == groundTruth[i]:
            count += 1
    return 100*count/len(y_predict)

if __name__ == "__main__":
    # Iris: tên 1 loài hoa, petal: cánh hoa, sepal: đài hoa
    iris = datasets.load_iris()
    iris_X = iris.data # data(petal length, petal width, sepal length, sepal width) matrix
    iris_y = iris.target # label

    # shuffle the array
    randIndex = np.arange(iris_X.shape[0])
    np.random.shuffle(randIndex)
    iris_X = iris_X[randIndex]
    iris_y = iris_y[randIndex]

    # chia dữ liệu để training và testing
    X_train = iris_X[:100]
    X_test = iris_X[100:]
    y_train = iris_y[:100]
    y_test = iris_y[100:]

    k = 5
    y_predict = []

    # assign label to each point
    for p in X_test:
        y_predict.append(predict(X_train, y_train, p, k))
    
    accuracy = accuracy_score(y_predict, y_test)
    print(accuracy,'%')