import mlflow
import mlflow.sklearn
import pytest
import pandas as pd
import numpy as np


# Fonction pour tester le chargement du modèle depuis MLflow
def test_model_loading():
    model_uri = "runs:/76a6e2e2f0544e7399c3a64ae086bfff/random_forest_model"
    model = mlflow.sklearn.load_model(model_uri)

    # on vérifie que le modèle est chargé correctement
    assert model is not None, "Le modèle n'a pas été chargé correctement"


# Fonction pour tester les prédictions du modèle
def test_model_prediction():
    # On charge le modèle depuis MLflow
    model_uri = "runs:/76a6e2e2f0544e7399c3a64ae086bfff/random_forest_model"
    model = mlflow.sklearn.load_model(model_uri)

    # données pour tester le modèle
    test_data = pd.DataFrame({
        'MedInc': [5.0],
        'HouseAge': [20.0],
        'AveRooms': [6.0],
        'AveBedrms': [2.0],
        'Population': [2000.0],
        'AveOccup': [3.0],
        'Latitude': [37.0],
        'Longitude': [-122.0]
    })

    #une prédiction avec le modèle
    prediction = model.predict(test_data)

    # Vérification que la prédiction a été effectuée
    assert prediction is not None, "La prédiction n'a pas été effectuée"
    assert isinstance(prediction, np.ndarray), "La prédiction doit être un tableau numpy"
    assert len(prediction) == 1, "La prédiction doit correspondre à un seul échantillon"
    assert prediction[0] > 0, "La prédiction doit être un nombre positif"
