import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np

# Charger les données
file_path = 'data/housing_standardized.csv'
data = pd.read_csv(file_path)

# Définir (features) et la variable cible (target)
X = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Début de l'expérience MLflow
mlflow.set_experiment("California Housing Gradient Boosting")

with mlflow.start_run(run_name="Gradient Boosting Model"):
    # Initialiser et entraîner le modèle Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
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

    # Enregistrer le modèle
    mlflow.sklearn.log_model(model, "gradient_boosting_model")
    print("Modèle enregistré dans MLflow.")
