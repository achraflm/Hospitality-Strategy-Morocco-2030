from sklearn.ensemble import AdaBoostRegressor

class AdaBoostModel:
    def __init__(self, random_state=42):
        self.model = AdaBoostRegressor(random_state=random_state)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X_test):
        return self.model.predict(X_test)
