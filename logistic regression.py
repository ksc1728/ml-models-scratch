# #--using sklearn--#
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

lr=LogisticRegression(max_iter=10000)

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,0,0,1])

lr.fit(X,y)
y_pred=lr.predict(X)
accuracy=np.sum(y==y_pred)/len(y)
print("Accuracy:",accuracy)


#---from scratch implementation---#
import numpy as np
class LogisicRegression:
    def __init__(self,epochs,lr):
        self.w=None
        self.b=None
        self.epochs=epochs
        self.lr=lr
    
    def sigmoid(self,z):
        return 1/ (1+ np.exp(-z))
    
    def fit(self,X,y):
        m,n=X.shape
        self.w=np.zeros(n)
        self.b=0
        for i in range(self.epochs):
            y_pred=self.sigmoid(np.dot(X,self.w)+self.b)
            dw=(1/m)*np.dot(X.T,(y_pred-y))
            db=(1/m)*np.sum(y_pred-y)
            self.w-=self.lr*dw
            self.b-=self.lr*db
        
    def predict(self,X):
        y_pred=np.dot(X,self.w)+self.b
        y_pred=self.sigmoid(y_pred)
        return (y_pred>=0.5).astype(int)

model=LogisicRegression(epochs=1000,lr=0.1)
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,1,1,1])
model.fit(X,y)
y_pred=model.predict(X)
accuracy=np.sum(y==y_pred)/len(y)
print("Accuracy:",accuracy)

new_X = np.array([[0,0], [1,1], [0,1], [0.2,0.3]])
new_pred = model.predict(new_X)
print("New Predictions:", new_pred)
