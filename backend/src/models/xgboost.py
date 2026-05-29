from xgboost import XGBRegressor

class XgboostModel:
    def __init__(self, random_state=42, verbosity=0):
        self.model = XGBRegressor(random_state=random_state, verbosity=verbosity)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X_test):
        return self.model.predict(X_test)
