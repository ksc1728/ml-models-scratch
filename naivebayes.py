#==using sklearn's Naive Bayes for comparison==#

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import numpy as np
iris=load_iris()
X=iris.data 
y=iris.target
model_nb=GaussianNB()
model_nb.fit(X,y)
y_pred=model_nb.predict(X)
accuracy=np.sum(y==y_pred)/len(y)
print("Sklearn Naive Bayes Accuracy:",accuracy)