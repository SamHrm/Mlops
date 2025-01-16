import pytest
import requests

API_URL = "http://127.0.0.1:8000/predict/"


@pytest.fixture
def housing_data():
    return {
        "MedInc": 6.0,
        "HouseAge": 15.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 37.0,
        "Longitude": -122.0
    }


def test_predict(housing_data):
    # Envoyer une requête POST à l'API pour obtenir une prédiction
    response = requests.post(API_URL, json=housing_data)

    # Vérifier que le code de statut de la réponse est 200 (OK)
    assert response.status_code == 200

    # Vérifier que la réponse contient une clé "predicted_house_value"
    response_data = response.json()
    assert "predicted_house_value" in response_data, "La réponse de l'API ne contient pas 'predicted_house_value'"

    # Vérifier que la valeur prédite est un nombre
    assert isinstance(response_data['predicted_house_value'], (int, float)), "La valeur prédite n'est pas un nombre"

