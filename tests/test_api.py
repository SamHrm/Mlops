import pytest
import requests

BASE_URL = "http://127.0.0.1:8000/predict/"  # FastAPI endpoint

# Test data (example)
data = {
    "MedInc": 8.3,
    "HouseAge": 41,
    "AveRooms": 6.98,
    "AveBedrms": 1.1,
    "Population": 300,
    "AveOccup": 2.5,
    "Latitude": 37.88,
    "Longitude": -122.23
}

# Define the test case
def test_fastapi_model_serving():
    headers = {"Content-Type": "application/json"}

    # Send the POST request to your FastAPI model server
    response = requests.post(BASE_URL, headers=headers, json=data)

    # Assert that the request was successful (status code 200)
    assert response.status_code == 200

    # Check if the predicted house value is in the response
    response_json = response.json()
    assert "predicted_house_value" in response_json

    # Check that the predicted house value is a float and positive
    assert isinstance(response_json["predicted_house_value"], float)
    assert response_json["predicted_house_value"] > 0
