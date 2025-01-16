import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

# Définir FastAPI
app = FastAPI()

model_uri = "runs:/76a6e2e2f0544e7399c3a64ae086bfff/random_forest_model"

# Charger le modèle depuis MLflow
model = mlflow.pyfunc.load_model(model_uri)


# Créer un modèle pour valider les entrées via FastAPI
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction du prix des maisons en Californie"}


@app.post("/predict/")
def predict(features: HousingFeatures):
    # Démarrer un nouveau run avec un nom personnalisé
    with mlflow.start_run(run_name="Prediction_Run"):
        # Convertir les données d'entrée en un tableau numpy
        input_data = np.array([[features.MedInc, features.HouseAge, features.AveRooms, features.AveBedrms,
                                features.Population, features.AveOccup, features.Latitude, features.Longitude]])

        # Faire une prédiction avec le modèle chargé
        prediction = model.predict(input_data)

        # Enregistrer la prédiction dans MLflow pour suivre
        mlflow.log_metric("prediction", prediction[0])  # Suivre la valeur de la prédiction

        # Retourner la prédiction sous forme de réponse JSON
        return {"predicted_house_value": prediction[0]}
