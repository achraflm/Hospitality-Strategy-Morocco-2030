from sklearn.ensemble import ExtraTreesRegressor

class ExtraTreesModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X_test):
        return self.model.predict(X_test)
