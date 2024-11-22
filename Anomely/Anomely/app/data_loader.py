import pandas as pd

def load_data(file_path):
    try:
        data=pd.read_csv(file_path)
        print("Model loaded successfully:", data)
        return data
    except Exception as e:
        raise ValueError("Error loading data from file: {e}")