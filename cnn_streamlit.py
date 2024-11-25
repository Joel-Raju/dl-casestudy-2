import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
import pickle


current_dir = os.getcwd()

model_path = os.path.join(
    current_dir,
    "cnn_model.h5",
)

model = keras.models.load_model(model_path)

label_encoder_path = os.path.join(
    current_dir,
    "cnn_model.h5",
)

# Load the label encoder from the pickle file
with open("cnn_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize the image to match the model's input shape
    img_array = (
        img_to_array(img) / 255.0
    )  # Normalize pixel values to be between 0 and 1

    img_array = np.expand_dims(
        img_array, axis=0
    )  # Add an extra dimension for batch size
    return img_array


# Streamlit app
st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(
        prediction
    )  # Get the index of the class with the highest probability

    # Decode the prediction using label_encoder (assuming you have a label_encoder defined)
    # Replace this with your actual label decoding logic
    predicted_label = label_encoder.classes_[predicted_class]

    # Display the prediction

    st.write(f"Predicted Class: **{predicted_label}**")
    st.write(f"Confidence Scores: {prediction[0]}")
