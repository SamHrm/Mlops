import shap
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend for saving plots

# Load data and model as before
file_path = 'data/housing_data.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']

model_uri = "runs:/76a6e2e2f0544e7399c3a64ae086bfff/random_forest_model"
model = mlflow.sklearn.load_model(model_uri)


explainer = shap.TreeExplainer(model)
X_subset = X.sample(500, random_state=42)  # we randomly select a subset of 500 rows from the dataset X to reduce Computation Time
shap_values_subset = explainer.shap_values(X_subset)

# Create SHAP plot
shap.summary_plot(shap_values_subset, X_subset)

# Save the plot as an image file
plt.savefig('shap_summary_plot.png', bbox_inches='tight', dpi=300)
print("SHAP summary plot saved to 'shap_summary_plot.png'.")


# Local SHAP analysis for an individual example
sample_index = 42  # Index of the row to analyze
sample = X.iloc[sample_index:sample_index + 1]  # Extract a single example

# Calculate SHAP values for the individual example
shap_values_sample = explainer.shap_values(sample)

# Create a force plot for the individual example
shap.force_plot(
    explainer.expected_value,  # Expected value is scalar for regression models
    shap_values_sample,        # SHAP values for the individual sample
    sample,
    matplotlib=True
)

# Save the force plot as an image
plt.savefig(f'shap_force_plot_sample_{sample_index}.png', bbox_inches='tight', dpi=300)
print(f"SHAP force plot saved to 'shap_force_plot_sample_{sample_index}.png'.")

# Display the SHAP values for the individual example
shap_values_df = pd.DataFrame(shap_values_sample, columns=sample.columns)
print(shap_values_df)
