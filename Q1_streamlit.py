import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import array_to_img
from Q1_DC_GAN_model import GAN


current_dir = os.getcwd()

model_path = os.path.join(
    current_dir,
    "q1_dcgan_model_final.keras",
)


# Streamlit app title
st.title("DCGAN Image Generator")
st.write("Generate images using a pretrained DCGAN model.")


# Load the GAN model
@st.cache_resource  # Cache the loaded model for faster inference
def load_gan_model(path):
    return load_model(path, custom_objects={"GAN": GAN})


gan = load_gan_model(model_path)
latent_dim = 128

# Slider to choose the number of images to generate
num_images = st.slider("Select number of images to generate:", 1, 10, 5)

# Button to trigger image generation
if st.button("Generate Images"):
    # Generate random latent vectors
    random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))

    # Generate fake images using the generator
    generated_images = gan.generator(random_latent_vectors)
    generated_images = (generated_images * 255).numpy().astype("uint8")  # Denormalize

    # Display the generated images
    st.write("Generated Images:")
    for i in range(num_images):
        st.image(array_to_img(generated_images[i]), caption=f"Image {i + 1}")
