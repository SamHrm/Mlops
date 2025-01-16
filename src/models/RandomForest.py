import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np

# Charger les données
file_path = 'data/housing_data.csv'
data = pd.read_csv(file_path)

# Définir (features) et la variable cible (target)
X = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Début de l'expérience MLflow
mlflow.set_experiment("California Housing Random Forest")

with mlflow.start_run(run_name="Random Forest Model") as run:
    # Initialiser et entraîner le modèle Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Racine carrée du MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Enregistrer les paramètres
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_estimators", 100)

    # Enregistrer les métriques
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R-squared", r2)

    # Afficher les métriques
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared: {r2}")

    # Enregistrer le modèle dans les artifacts
    model_name = "random_forest_model"
    mlflow.sklearn.log_model(model, model_name)
    print("Modèle enregistré dans les artifacts de MLflow.")

    # Enregistrer dans le Model Registry
    registered_model_name = "CaliforniaHousingRandomForest"
    model_uri = f"runs:/{run.info.run_id}/{model_name}"
    mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    print(f"Modèle enregistré dans le Model Registry avec le nom : {registered_model_name}")
