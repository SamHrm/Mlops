import pytest
from fastapi.testclient import TestClient
from API.main import app  # Assuming your FastAPI app is in API.main

@pytest.fixture
def client():
    """Fixture to create a TestClient for FastAPI"""
    return TestClient(app)

@pytest.fixture
def housing_data():
    """Fixture to provide example housing data"""
    return {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }

def test_predict(client, housing_data):
    """Test the /predict/ endpoint of the API"""

    # Send POST request to /predict/ endpoint with input data
    response = client.post("/predict/", json=housing_data)

    # Check if the request was successful
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"

    # Check if the response contains the predicted house value
    response_data = response.json()
    assert "predicted_house_value" in response_data, "Response should contain 'predicted_house_value' key"
    assert isinstance(response_data['predicted_house_value'], (int, float)), "Predicted value should be a number"
