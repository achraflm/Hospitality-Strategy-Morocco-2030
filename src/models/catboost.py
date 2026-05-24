from catboost import CatBoostRegressor

class CatboostModel:
    def __init__(self, random_state=42, verbose=False):
        self.model = CatBoostRegressor(random_state=random_state, verbose=verbose)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X_test):
        return self.model.predict(X_test)
