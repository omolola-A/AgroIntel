import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import json
import openai

# Load the model
model = load_model(r"C:\Users\Dell\Desktop\3MTT\Hackaton 2.0\Model_training\cnn_model.keras")

# Load class labels
with open(r"C:\Users\Dell\Desktop\3MTT\Hackaton 2.0\Model_training\class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = list(class_indices.keys())

# Set OpenAI API key
openai.api_key = "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"  # Replace with your actual API key

# Function to get recommendations from ChatGPT
def get_recommendations(predicted_class):
    prompt = f"""
    I am building a crop disease and pest detection app for farmers. 
    Based on the prediction "{predicted_class}", provide actionable advice for the farmer to handle this issue effectively.
    Please keep the advice concise and practical.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert agricultural advisor."},
                {"role": "user", "content": prompt}
            ]
        )
        advice = response['choices'][0]['message']['content'].strip()
        return advice
    except Exception as e:
        return f"Error generating recommendations: {e}"
    
# App Title
st.title("Crop Pest and Disease Prediction App with Recommendations")

# File uploader
uploaded_file = st.file_uploader("Upload a Crop Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show result
    st.write(f"Prediction: **{predicted_class}**")

    # Get recommendations
    advice = get_recommendations(predicted_class)
    st.write(f"Recommendations: {advice}")
