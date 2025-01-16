import streamlit as st
import requests

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000/predict/"

# Fonction pour envoyer les données à l'API et récupérer la prédiction
def get_prediction(features):
    response = requests.post(API_URL, json=features)
    if response.status_code == 200:
        return response.json()['predicted_house_value']
    else:
        st.error(f"Erreur lors de la prédiction: {response.status_code}")
        return None

# Interface Streamlit

st.title("Prédiction du prix des maisons en Californie")
st.write("Entrez les caractéristiques de la maison pour obtenir une prédiction du prix.")

# Champs pour entrer les caractéristiques de la maison
MedInc = st.number_input("Revenu médian des ménages (MedInc)", min_value=0.0)
HouseAge = st.number_input("Âge de la maison (HouseAge)", min_value=0.0)
AveRooms = st.number_input("Nombre moyen de chambres (AveRooms)", min_value=0.0)
AveBedrms = st.number_input("Nombre moyen de chambres à coucher (AveBedrms)", min_value=0.0)
Population = st.number_input("Population de la zone (Population)", min_value=0.0)
AveOccup = st.number_input("Nombre moyen d'occupants (AveOccup)", min_value=0.0)
Latitude = st.number_input("Latitude de la maison (Latitude)", min_value=-90.0, max_value=90.0, step=0.01)
Longitude = st.number_input("Longitude de la maison (Longitude)", min_value=-180.0, max_value=180.0, step=0.01)

# Lorsque l'utilisateur clique sur le bouton pour prédire
if st.button("Prédire le prix"):
    features = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }

    # Faire la prédiction
    prediction = get_prediction(features)

    if prediction is not None:
        st.success(f"Le prix prédit de la maison est : ${prediction:.2f}")
