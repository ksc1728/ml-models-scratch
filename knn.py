#---using sklearn library----

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()
X=iris.data
y=iris.target
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
y_pred=knn.predict(X)
accuracy=np.sum(y==y_pred)/len(y)
print("Sklearn KNN Accuracy:",accuracy)

#---from scratch implementation of k-nearest neighbors algorithm---#
import numpy as np
from collections import Counter
class KNNScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train=X
        self.y_train=y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return np.bincount(k_nearest_labels).argmax()

X_train = np.array([
    [1.0, 1.2],  # Class 0
    [1.5, 1.8],  # Class 0
    [5.0, 8.0],  # Class 1
    [8.0, 8.2]   # Class 1
])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([
    [1.1, 1.5],  # Should be predicted as Class 0
    [7.5, 9.0]   # Should be predicted as Class 1
])


knn = KNNScratch(k=2) 

knn.fit(X_train, y_train)
print("Model trained with data:\n", "X_train:\n", X_train, "\ny_train:", y_train)

predictions = knn.predict(X_test)

print("\n--- Results ---")
print("Test Data (X_test):\n", X_test)
print("Predictions (k=1):", predictions)