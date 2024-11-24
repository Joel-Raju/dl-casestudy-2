import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the VAE model
vae_model = load_model("vae_final_model.h5")


# Function to generate an image from the model
def generate_image(latent_dim):
    """Generates an image from the VAE model given a latent vector."""
    random_latent_vector = np.random.normal(size=(1, latent_dim))
    generated_image = vae_model.decoder.predict(random_latent_vector)
    return generated_image[0]


# Streamlit app
st.title("VAE Image Generator")

# Slider for latent dimension input
latent_dim = st.slider(
    "Latent Dimension", min_value=32, max_value=256, value=128, step=32
)

# Button to generate image
if st.button("Generate Image"):
    generated_image = generate_image(latent_dim)
    st.image(generated_image, caption="Generated Image", use_column_width=True)
