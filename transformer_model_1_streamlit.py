import streamlit as st
import tensorflow as tf
import os
import re
from transformer_model_1 import (
    TokenAndPositionEmbedding,
    weighted_categorical_crossentropy,
    TransformerBlock,
)


current_dir = os.getcwd()

model_path = os.path.join(
    current_dir,
    "transformer_model_1_neutral_enhanced.h5",
)

# Load the model with the required custom objects
loaded_model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,
        "weighted_categorical_crossentropy": weighted_categorical_crossentropy,
    },
)


# Streamlit app
st.title("Transformer Model Tester")

# Text input
text = st.text_area("Enter text:", "Example text")


# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    # Tokenize (replace with your actual tokenizer)
    # For example, using simple space-based tokenization:
    tokens = text.split()
    # Pad the sequence (replace with your actual padding logic)
    # ...
    return tokens  # Or your padded/encoded sequence


# Predict button
if st.button("Predict"):
    # Preprocess the text
    processed_input = preprocess_text(text)

    # Make prediction
    prediction = loaded_model.predict(processed_input)

    # Display the prediction
    st.write("Prediction:", prediction)
