import numpy as np

class MyLinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.b0=0
        self.b1=0
        
    def fit(self, X, y):
        n=len(X)
        
        for _ in range(self.n_iters):
            y_pred=self.b0+self.b1*X
            
            # Gradients
            db0=(-2/n)*np.sum(y-y_pred)
            db1=(-2/n)*np.sum(X*(y-y_pred))
            
            #update
            self.b0-=self.lr*db0
            self.b1-=self.lr*db1
            
    def predict(self, X):
        return self.b0+self.b1*X
    


X=np.array([1, 2, 3, 4, 5])
y=np.array([2, 4, 5, 4 ,5 ])

model=MyLinearRegression(lr=0.01, n_iters=1000)
model.fit(X, y)

predictions=model.predict(X)
print(predictions)

