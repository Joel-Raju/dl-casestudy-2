import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim):
    encoder_inputs = tf.keras.layers.Input(shape=(64, 64, 3))

    # Convolutional layers
    x = tf.keras.layers.Conv2D(32, 4, strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Flatten and Dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Output layers
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def build_decoder(latent_dim):
    decoder_inputs = tf.keras.layers.Input(shape=(latent_dim,))

    # Dense layers
    x = tf.keras.layers.Dense(512)(decoder_inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dense(4 * 4 * 256)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Reshape((4, 4, 256))(x)

    # Transposed convolution layers
    x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(16, 4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Output layer
    decoder_outputs = tf.keras.layers.Conv2D(
        3, 3, padding="same", activation="sigmoid"
    )(x)

    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")
    return decoder


class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = build_encoder(latent_dim)
        self.decoder = build_decoder(latent_dim)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Get encoder outputs
            z_mean, z_log_var, z = self.encoder(data)
            # Get reconstruction
            reconstruction = self.decoder(z)

            # Compute reconstruction loss - MSE instead of binary crossentropy
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=[1, 2, 3])
            )

            # Compute KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
                )
            )

            # Total loss with beta-VAE weighting
            beta = 1.0  # Adjust this value to control disentanglement
            total_loss = reconstruction_loss + beta * kl_loss

        # Compute and apply gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


current_dir = os.getcwd()

model_path = os.path.join(
    current_dir, "vae_best_model.keras", custom_objects={"VAE": VAE}
)

# Load the VAE model
vae_model = load_model(model_path)


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
