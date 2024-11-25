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


# Lists of strong positive and negative words to reduce their weight in Neutral reviews
positive_words = [
    "amazing",
    "fantastic",
    "great",
    "excellent",
    "love",
    "wonderful",
    "perfect",
]
negative_words = ["terrible", "horrible", "awful", "hate", "bad", "worst", "disgusting"]

# Phrases commonly associated with Neutral sentiments
neutral_phrases = {
    "it was okay": "neutral",
    "not bad, but not great": "neutral",
    "average experience": "neutral",
    "decent enough": "neutral",
    "nothing special": "neutral",
    "just fine": "neutral",
    # Add more phrases as needed
}


# Reduce strong sentiment words in Neutral reviews
def reduce_sentiment_intensity(text, strong_words):
    words = text.split()
    return " ".join(["neutral" if word in strong_words else word for word in words])


# Handle common neutral phrases and synonym replacements
def handle_neutral_patterns(text):
    # Replace known neutral phrases
    for phrase, replacement in neutral_phrases.items():
        text = re.sub(
            r"\b" + re.escape(phrase) + r"\b", replacement, text, flags=re.IGNORECASE
        )

    # Synonym replacement for toned-down words
    words = text.split()
    toned_down_words = []
    for word in words:
        if word in positive_words:
            toned_down_words.append("decent")
        elif word in negative_words:
            toned_down_words.append("not ideal")
        else:
            toned_down_words.append(word)
    return " ".join(toned_down_words)


# Integrate into enhanced preprocessing function
def enhanced_neutral_preprocess_text(text):
    # Basic cleanup
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)  # Remove special characters
    text = text.lower()

    # Reduce strong sentiment words
    text = reduce_sentiment_intensity(text, positive_words + negative_words)

    # Handle specific neutral patterns and toned-down synonym replacements
    text = handle_neutral_patterns(text)

    return text


# Streamlit app
st.title("Transformer Model Tester")

# Text input
text = st.text_area("Enter text:", "Example text")

# Predict button
if st.button("Predict"):
    # Preprocess the text
    processed_input = enhanced_neutral_preprocess_text(text)

    # Make prediction
    prediction = loaded_model.predict(processed_input)

    # Display the prediction
    st.write("Prediction:", prediction)
