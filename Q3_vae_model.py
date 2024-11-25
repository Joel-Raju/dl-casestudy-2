import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable
import os
import logging
import numpy as np


@register_keras_serializable()
class VAE(Model):
    def __init__(self, latent_dim, input_shape=(128, 128, 3), **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Initialize encoder and decoder
        self._build_encoder()
        self._build_decoder()

        # Track losses
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def _build_encoder(self):
        encoder_inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(
            encoder_inputs
        )
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        self.encoder = Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

    def _build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(16 * 16 * 128, activation="relu")(latent_inputs)
        x = layers.Reshape((16, 16, 128))(x)
        x = layers.Conv2DTranspose(
            128, 3, strides=2, padding="same", activation="relu"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(
            x
        )
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(
            x
        )
        x = layers.BatchNormalization()(x)
        decoder_outputs = layers.Conv2DTranspose(
            3, 3, strides=1, padding="same", activation="sigmoid"
        )(x)

        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    def reparameterize(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_config(self):
        # Return configuration for serialization
        config = super(VAE, self).get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "input_shape": self.input_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Create an instance from the serialized configuration
        return cls(**config)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Forward pass
            reconstruction, z_mean, z_log_var = self(data, training=True)
            # Flatten data and reconstruction to compute binary crossentropy
            flattened_data = tf.reshape(data, [-1, np.prod(self.input_shape)])
            flattened_reconstruction = tf.reshape(
                reconstruction, [-1, np.prod(self.input_shape)]
            )
            # Compute reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    y_true=flattened_data, y_pred=flattened_reconstruction
                )
            )
            # Compute KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            # Total loss
            total_loss = reconstruction_loss + kl_loss
        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update losses
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def preprocess_image_batch(image_paths, target_size=(128, 128)):
    """Preprocess a batch of images for VAE input."""

    def process_single_image(image_path):
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, target_size)
            image = tf.cast(image, tf.float32) / 255.0
            return image
        except tf.errors.InvalidArgumentError:
            return tf.zeros(target_size + (3,), dtype=tf.float32)

    images = tf.map_fn(process_single_image, image_paths, dtype=tf.float32)
    return images


def create_dataset(folder_path, batch_size=64, buffer_size=1000):
    """Create tensorflow dataset with optimized batching for GPU."""
    if not os.path.exists(folder_path):
        raise ValueError(f"Directory not found: {folder_path}")

    # Optimize the input pipeline
    dataset = (
        tf.data.Dataset.list_files(f"{folder_path}/*.png", shuffle=True)
        .batch(batch_size)  # Batch the file paths first
        .map(
            preprocess_image_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .cache()  # Cache the preprocessed images
        .shuffle(buffer_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


def train_vae(train_dir, epochs=50, batch_size=256, latent_dim=128, learning_rate=1e-4):
    """Main training function."""
    strategy = setup_strategy()

    with strategy.scope():
        # Initialize model, optimizer and dataset
        vae = VAE(latent_dim=latent_dim)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        train_dataset = create_dataset(train_dir, batch_size)

        checkpoint_dir = "checkpoints_dir"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Train the model
        history = vae.fit(
            train_dataset,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(checkpoint_dir, "vae_checkpoint_{epoch:02d}.keras"),
                    save_best_only=True,
                    monitor="loss",
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss", patience=3, restore_best_weights=False
                ),
            ],
        )

        return vae, history
