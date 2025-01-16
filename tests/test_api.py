import pytest
from fastapi.testclient import TestClient
from API.main import app  # Assuming your FastAPI app is in API.main

@pytest.fixture
def client():
    """Fixture to create a TestClient for FastAPI"""
    return TestClient(app)

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
