import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

# Assume the FastAPI app is in a file called 'main.py'
from API.main import app  # Make sure the FastAPI app is imported from the correct file

# Set the MLflow tracking URI (if it's a remote server, make sure it's set to the correct address)
# For local usage, this is not necessary if it's running locally with the default URI.
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# MLflow model URI (from the run ID on the MLflow server)
model_uri = 'runs:/76a6e2e2f0544e7399c3a64ae086bfff/random_forest_model'

# Load the model from MLflow (make sure the server is running and the model exists)
model = mlflow.pyfunc.load_model(model_uri)

# Create the TestClient instance to simulate HTTP requests to your FastAPI app
client = TestClient(app)


# Test data for the HousingFeatures model
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# Test the root endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de prédiction du prix des maisons en Californie"}


# Test the prediction endpoint
def test_predict():
    # Prepare test data for the housing features
    test_data = HousingFeatures(
        MedInc=8.3252,
        HouseAge=41.0,
        AveRooms=6.984126984126984,
        AveBedrms=1.0238095238095237,
        Population=322.0,
        AveOccup=2.5555555555555554,
        Latitude=37.88,
        Longitude=-122.23
    )

    # Send a POST request to the /predict/ endpoint with the test data
    response = client.post("/predict/", json=test_data.dict())

    # Check that the response is valid and contains the predicted value
    assert response.status_code == 200
    assert "predicted_house_value" in response.json()


# Test invalid input (missing required fields)
def test_invalid_input():
    # Send a POST request with invalid data (e.g., missing some required fields)
    invalid_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984126984126984,
        "AveBedrms": 1.0238095238095237
        # Missing other required fields like Population, Latitude, etc.
    }

    response = client.post("/predict/", json=invalid_data)

    # Ensure that the response contains an error due to invalid input
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()
