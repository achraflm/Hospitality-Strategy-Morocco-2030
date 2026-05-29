from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class SvrModel:
    def __init__(self):
        self.model = SVR(kernel='rbf', C=100.0, epsilon=0.1)
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self
        
    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
