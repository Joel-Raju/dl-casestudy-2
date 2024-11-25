import streamlit as st
import tensorflow as tf
from tensorflow import keras
import os
from Q1_DC_GAN_model import GAN


current_dir = os.getcwd()

model_path = os.path.join(
    current_dir,
    "Q1_DC_GAN.keras",
)

# Load the trained GAN model
gan = keras.models.load_model(model_path, custom_objects={"GAN": GAN})


# Define the Streamlit app
def main():
    st.title("GAN Image Generator")
    st.sidebar.header("Settings")

    # Number of images to generate
    num_images = st.sidebar.slider("Number of Images", 1, 10, 3)

    # Latent dimension size (should match your model)
    latent_dim = 128  # Update if your model uses a different latent dim

    if st.button("Generate Images"):
        # Generate random latent vectors
        random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))

        # Generate images
        generated_images = gan.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()

        # Display images
        for i in range(num_images):
            img = keras.utils.array_to_img(generated_images[i])
            st.image(img, caption=f"Generated Image {i+1}", use_column_width=True)


if __name__ == "__main__":
    main()
