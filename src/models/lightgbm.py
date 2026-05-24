from lightgbm import LGBMRegressor

class LightgbmModel:
    def __init__(self, random_state=42, verbose=-1):
        self.model = LGBMRegressor(random_state=random_state, verbose=verbose)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X_test):
        return self.model.predict(X_test)
