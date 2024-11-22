import joblib
import os
from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetector:
    def __init__(self, model_path="models/isolation_forest.pkl"):
       
        self.model_path = model_path
        self.model = None  

    def train(self, data):
       
        X = np.array(data)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)  
        
       
        self.model = IsolationForest(random_state=42, contamination=0.1)
        self.model.fit(X)

       
        self.save_model()
        print("Model trained and saved successfully.")

    def predict(self, data):
       
 
        if self.model is None:
            self.model = self.load_model()
        
        if self.model is None:
            raise ValueError("Model is not loaded. Train or load a model first.")
        
        X = np.array(data)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)  
        
        
        return self.model.predict(X)

    def save_model(self):
        
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                print("Model loaded successfully:", model)
                print(f"Model loaded from {self.model_path}")
                return model
            except Exception as e:
                print(f"Failed to load model: {e}")
                return None
        print(f"No pre-trained model found at {self.model_path}")
        return None
