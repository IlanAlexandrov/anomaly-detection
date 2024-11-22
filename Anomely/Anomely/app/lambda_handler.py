import json
import boto3
from app.anomaly_detector import AnomalyDetector
from app.data_loader import load_data
import os

# AWS S3 client
s3 = boto3.client('s3')

def lambda_train_handler(event, context):
    """
    AWS Lambda handler for training the model.
    Saves the trained model to S3.
    """
    body = json.loads(event['body'])
    client_id = body.get('client_id')
    file_path = body.get('file_path')
    bucket_name = "your-s3-bucket-name"  # Replace with your S3 bucket name

    if not client_id or not file_path:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "client_id and file_path are required"})
        }

    try:
        model_path = f"/tmp/{client_id}_model.pkl"
        data = load_data(file_path)
        detector = AnomalyDetector(model_path=model_path)
        detector.train(data.values)
        s3.upload_file(model_path, bucket_name, f"models/{client_id}_model.pkl")

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Model trained and uploaded to S3 successfully"})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

def lambda_predict_handler(event, context):
    """
    AWS Lambda handler for making predictions.
    Loads the trained model from S3.
    """
    body = json.loads(event['body'])
    client_id = body.get('client_id')
    prediction_data = body.get('data')
    bucket_name = "your-s3-bucket-name"  # Replace with your S3 bucket name

    if not client_id or not prediction_data:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "client_id and data are required"})
        }

    try:
        model_path = f"/tmp/{client_id}_model.pkl"
        s3.download_file(bucket_name, f"models/{client_id}_model.pkl", model_path)
        detector = AnomalyDetector(model_path=model_path)
        predictions = detector.predict(prediction_data)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "client_id": client_id,
                "predictions": predictions.tolist()
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

# Testing Block
if __name__ == "__main__":
    # Mock event for training
    train_event = {
        "body": json.dumps({
            "client_id": "client1",
            "file_path": "C:\\ilan\\anomal\\sample_test_data.csv"
        })
    }
    print(lambda_train_handler(train_event, None))

    # Mock event for prediction
    predict_event = {
        "body": json.dumps({
            "client_id": "client1",
            "data": [[55], [45], [75], [50]]
        })
    }
    print(lambda_predict_handler(predict_event, None))
    