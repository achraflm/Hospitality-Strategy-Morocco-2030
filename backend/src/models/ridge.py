from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class RidgeModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Fits Ridge model."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self
        
    def predict(self, X_test):
        """Predicts using Ridge model."""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
