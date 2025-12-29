#-----using sklearn library-----#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X=np.random.rand(50,1)*100
Y=3.5*X+np.random.rand(50,1)*20


model=LinearRegression()
model.fit(X,Y)

Y_pred=model.predict(X)
plt.scatter(X,Y,color='b')
plt.plot(X,Y_pred,color='r')
plt.show()


#from scratch implementation
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=None 
        self.bias=None
    
    def fit(self,X,y):
        m,n=X.shape
        self.weights=np.zeros(n)
        self.bias=0
        
        for i in range(self.epochs):
            # Hypothesis: h(X) = Xw + b
            y_pred=np.dot(X,self.weights)+self.bias
            
            # Gradient descent
            # Gradient of Cost J(w,b) w.r.t weights (dw) and bias (db)
            dw=(1/m)*np.dot(X.T,(y_pred - y))
            db=(1/m)*np.sum(y_pred - y)
            
            # Update parameters
            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db
            
    def predict(self,X):
        y_pred=np.dot(X,self.weights)+self.bias
        return y_pred        

# --- Data Generation and Model Training ---
np.random.seed(42)
# X is (100, 5) - 100 samples, 5 features
X = np.random.rand(100, 5) 
# y is (100,) - 100 targets
y = 2 * X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100) 

lr=LinearRegression(learning_rate=0.1, epochs=5000) 
lr.fit(X,y)
y_pred=lr.predict(X)

# --- Visualization (Predictions vs. Actual) ---
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='b', alpha=0.6, label='Predicted vs Actual')
# Plotting the ideal line y=x
plt.plot([min(y), max(y)], [min(y), max(y)], color='r', linestyle='--', linewidth=2, label='Ideal Fit ($y=\hat{y}$)') 
plt.xlabel("Actual Values ($y$)", fontsize=14)
plt.ylabel("Predicted Values ($\hat{y}$)", fontsize=14)
plt.title("Linear Regression: Actual vs. Predicted Values", fontsize=16)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.show()
