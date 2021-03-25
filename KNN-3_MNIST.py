import numpy as np
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    digit = datasets.load_digits()
    digit_X = digit.data
    digit_y = digit.target

    # shuffle the array
    randIndex = np.arange(digit_X.shape[0])
    np.random.shuffle(randIndex)
    digit_X = digit_X[randIndex]
    digit_y = digit_y[randIndex]

    X_train, X_test, y_train, y_test = train_test_split(digit_X, digit_y, test_size = 360)

    # use sklearn library
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    y_predict = knn.predict(X_test)

    accuracy = accuracy_score(y_predict, y_test)
    print(accuracy)

    plt.gray()
    plt.imshow(X_test[0].reshape((8,8)))
    print(knn.predict(X_test[0].reshape(1,-1)))
    plt.show()