from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np

iris=load_iris()
X=iris.data 
y=iris.target
model_svm=SVC(kernel='rbf',C=1.0)
model_svm.fit(X,y)
y_pred=model_svm.predict(X)
accuracy=np.sum(y==y_pred)/len(y)
print("Sklearn SVM Accuracy:",accuracy)