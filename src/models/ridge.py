from sklearn.linear_model import Ridge

class RidgeModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        
    def fit(self, X_train, y_train):
        """Fits Ridge model."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X_test):
        """Predicts using Ridge model."""
        return self.model.predict(X_test)
