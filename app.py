import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np
import os

# Model dosyasının bulunduğu klasör
MODEL_PATH = "/Users/musticodes/Desktop/diabetes"

@st.cache_data
def load_model_and_metadata(path):
    with open(os.path.join(path, 'diabetes_model.pkl'), 'rb') as file:
        model = pickle.load(file)

    with open(os.path.join(path, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)

    return model, metadata

def main():
    st.title("Diabetes Prediction App with Random Forest")
    st.write("📊 Bu uygulama daha önce eğitilmiş bir modeli kullanarak diyabet tahmini yapar.")

    model, metadata = load_model_and_metadata(MODEL_PATH)

    st.subheader("🔎 Model Bilgileri")
    st.json({
        "Model Adı": metadata["model_name"],
        "Oluşturulma Tarihi": metadata["created_date"],
        "Performans Metrikleri": metadata["performance_metrics"],
        "Feature Sayısı": len(metadata["feature_names"]),
    })

    st.subheader("🧪 Model ile Tahmin Yap")

    input_data = {}
    for feature in metadata["feature_names"]:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Tahmin Et"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        st.write(f"🔍 Tahmin: **{'Diyabetli' if prediction == 1 else 'Diyabetli Değil'}**")
        st.write(f"📈 Olasılık (pozitif sınıf): `{prediction_proba:.2f}`")

    st.subheader("⚙️ Hiperparametreler")
    if st.checkbox("Hiperparametreleri Göster"):
        st.json(metadata["hyperparameters"])

if __name__ == "__main__":
    main()
