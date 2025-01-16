import pytest
from fastapi.testclient import TestClient
from API.main import app  # Import your FastAPI app

# Create a test client for the FastAPI app
client = TestClient(app)

@pytest.fixture
def valid_input():
    """Fixture to provide valid input data for predictions."""
    return {
        "MedInc": 3.5,
        "HouseAge": 20,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1500,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }

@pytest.fixture
def invalid_input():
    """Fixture to provide invalid input data for predictions."""
    return {
        "MedInc": "invalid",  # Invalid type
        "HouseAge": 20,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1500,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25
    }

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200, "Root endpoint should return HTTP 200"
    assert "message" in response.json(), "Response should contain a 'message' key"

def test_predict_valid_input(valid_input):
    """Test the /predict/ endpoint with valid input."""
    response = client.post("/predict/", json=valid_input)
    assert response.status_code == 200, "Predict endpoint should return HTTP 200 for valid input"
    response_data = response.json()
    assert "predicted_house_value" in response_data, "Response should contain 'predicted_house_value'"
    assert isinstance(response_data["predicted_house_value"], (int, float)), "Prediction should be a number"

def test_predict_invalid_input(invalid_input):
    """Test the /predict/ endpoint with invalid input."""
    response = client.post("/predict/", json=invalid_input)
    assert response.status_code == 422, "Predict endpoint should return HTTP 422 for invalid input"

def test_predict_missing_fields():
    """Test the /predict/ endpoint with missing fields."""
    incomplete_input = {
        "MedInc": 3.5,
        "HouseAge": 20,
        "AveRooms": 6.0
        # Missing other fields
    }
    response = client.post("/predict/", json=incomplete_input)
    assert response.status_code == 422, "Predict endpoint should return HTTP 422 for missing fields"

def test_predict_edge_cases():
    """Test the /predict/ endpoint with edge case inputs."""
    edge_case_input = {
        "MedInc": 0.0,  # Minimum income
        "HouseAge": 100,  # Oldest house
        "AveRooms": 1.0,  # Minimum rooms
        "AveBedrms": 1.0,
        "Population": 1,  # Minimum population
        "AveOccup": 1.0,
        "Latitude": -90.0,  # Minimum valid latitude
        "Longitude": -180.0  # Minimum valid longitude
    }
    response = client.post("/predict/", json=edge_case_input)
    assert response.status_code == 200, "Predict endpoint should handle edge cases gracefully"
    response_data = response.json()
    assert "predicted_house_value" in response_data, "Response should contain 'predicted_house_value'"
