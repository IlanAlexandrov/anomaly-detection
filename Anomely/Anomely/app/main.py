import sys
import os
print(sys.path)
from flask import Flask, request, jsonify
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.anomaly_detector import AnomalyDetector
from app.data_loader import load_data



app = Flask(__name__)


BASE_MODEL_DIR = "models"

@app.route('/train', methods=['POST'])  
def train_model():
    """
    Endpoint to train the model using a client-specific dataset.
    """
    file_path = request.json.get('file_path')
    client_id = request.json.get('client_id')
    if not file_path or not client_id:
        return jsonify({"error": "File path and client_id are required"}), 400

    try:
        
        os.makedirs(BASE_MODEL_DIR, exist_ok=True)

        
        client_model_path = os.path.join(BASE_MODEL_DIR, f"{client_id}_model.pkl")

        
        data = load_data(file_path)
        detector = AnomalyDetector(model_path=client_model_path)
        detector.train(data.values)

        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict anomalies using a client-specific model.
    """
    client_id = request.json.get('client_id')
    prediction_data = request.json.get('data')
    if not client_id or not prediction_data:
        return jsonify({"error": "client_id and data are required"}), 400

    try:
        
        client_model_path = os.path.join(BASE_MODEL_DIR, f"{client_id}_model.pkl")

        
        detector = AnomalyDetector(model_path=client_model_path)

        
        predictions = detector.predict(prediction_data)
        return jsonify({
            "client_id": client_id,
            "predictions": predictions.tolist()
        }), 200
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Please train the model first."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run on default Flask port (5000)
    app.run(debug=True)

    # Uncomment to run on a different port
    # app.run(debug=True, host='0.0.0.0', port=8080)
