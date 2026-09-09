import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr=lr
        self.lambda_param=lambda_param
        self.n_iters=n_iters
        self.w=None
        self.b=None
        
    def fit(self, X, y):
        n_samples, n_features=X.shape
        
        # Convert labels to -1 and 1
        y_=np.where(y<= 0, -1, 1)
        
        self.w=np.zeros(n_features)
        self.b=0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition=y_[idx]*(np.dot(x_i, self.w)+self.b)>=1
                
                if condition:
                    #only regularization term
                    self.w-=self.lr*(2*self.lambda_param*self.w)
                    
                else:
                    #Misclassified
                    self.w-=self.lr*(2*self.lambda_param*self.w-np.dot(x_i, y_[idx]))
                    self.b-=self.lr*y_[idx]
                    
    def predict(self, X):
        linear_output=np.dot(X, self.w)+self.b   #type:ignore
        return np.sign(linear_output)
    
    
class MultiClassSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.models=[]
        self.lr=lr
        self.lambda_param=lambda_param
        self.n_iters=n_iters
        
    def fit(self, X, y):
        self.classes=np.unique(y)
        
        for c in self.classes:
            y_binary=np.where(y==c, 1, -1)
            
            model=SVM(self.lr, self.lambda_param, self.n_iters)
            model.fit(X, y_binary)
            
            self.models.append(model)
            
    def predict(self, X):
        scores=[]
        
        for model in self.models:
            score=np.dot(X, model.w)+model.b
            scores.append(score)
            
        scores=np.array(scores)
        predictions=np.argmax(scores, axis=0)
        return self.classes[predictions]
    
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

digits=load_digits()
X,y=digits.data, digits.target #type:ignore

#Split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Scale
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_train)

#Train
model=MultiClassSVM(lr=0.001, lambda_param=0.01, n_iters=1000)
model.fit(X_train, y_train)

#Predict
y_pred=model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))