import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import array_to_img
import matplotlib.pyplot as plt

# Streamlit app title
st.title("GAN Image Generator")

# Load the GAN model
@st.cache_resource
def load_gan_model():
    model = load_model("gan_final_model.h5")
    return model

generator = load_gan_model()

# User input: Number of images to generate
num_images = st.slider("Number of images to generate", min_value=1, max_value=10, value=5, step=1)

# Latent dimension (update this if your model uses a different latent dimension)
latent_dim = 100  # Example value; adjust as per your GAN

# Generate images
if st.button("Generate Images"):
    st.write(f"Generating {num_images} images...")

    # Generate random latent vectors
    random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))

    # Use the generator to create fake images
    generated_images = generator(random_latent_vectors)

    # Denormalize the images to bring them back to [0, 255] range
    generated_images = (generated_images * 255).numpy().astype("uint8")

    # Display images
    st.write("Generated Images:")
    cols = st.columns(num_images)
    for i in range(num_images):
        with cols[i]:
            st.image(array_to_img(generated_images[i]), caption=f"Image {i + 1}", use_column_width=True)
