import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Path to data
file_path = 'data/housing_data.csv'

@pytest.fixture
def load_data():
    """Fixture to load the dataset."""
    data = pd.read_csv(file_path)
    return data

def test_data_loading(load_data):
    """Test if data loads correctly."""
    data = load_data
    assert not data.empty, "The dataset is empty"
    assert 'MedHouseVal' in data.columns, "Target column 'MedHouseVal' is missing"
    assert len(data.columns) > 1, "Insufficient feature columns"

def test_train_test_split(load_data):
    """Test if train-test split works correctly."""
    data = load_data
    X = data.drop(columns=['MedHouseVal'])
    y = data['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) / len(data) == pytest.approx(0.8, 0.01), "Train set size is incorrect"
    assert len(X_test) / len(data) == pytest.approx(0.2, 0.01), "Test set size is incorrect"

def test_model_training(load_data):
    """Test if the Random Forest model trains without errors."""
    data = load_data
    X = data.drop(columns=['MedHouseVal'])
    y = data['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    assert hasattr(model, "predict"), "Model training failed; 'predict' method missing"

def test_metric_calculations(load_data):
    """Test if the metrics are calculated correctly."""
    data = load_data
    X = data.drop(columns=['MedHouseVal'])
    y = data['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    assert mse >= 0, "MSE should be non-negative"
    assert rmse >= 0, "RMSE should be non-negative"
    assert mae >= 0, "MAE should be non-negative"
    assert -1 <= r2 <= 1, "R2 should be between -1 and 1"

def test_model_predictions_shape(load_data):
    """Test if the predictions have the correct shape."""
    data = load_data
    X = data.drop(columns=['MedHouseVal'])
    y = data['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test), "Predictions shape mismatch"