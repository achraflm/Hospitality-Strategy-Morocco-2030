from sklearn.svm import SVR

class SvrModel:
    def __init__(self):
        self.model = SVR()
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X_test):
        return self.model.predict(X_test)
