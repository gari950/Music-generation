import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Conv1D, Flatten, Reshape, Conv1DTranspose, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import music21
import os

midi_directory = 'C:\\Users\\garima saha\\GAN\\midi_songs\\'

list_of_midi_files = [os.path.join(midi_directory, file) for file in os.listdir(midi_directory) if file.endswith('.midi')]

# Helper function to convert MIDI files to note sequences
def midi_to_note_sequence(midi_file_path, sequence_length=128):
    midi_data = music21.converter.parse(midi_file_path)
    notes = []
    for element in midi_data.flatten().notes:
        pitch = element.pitches
        duration = element.duration.quarterLength
        notes.append((pitch, duration))
    return notes[:sequence_length]

# Function to get real music samples
def get_real_samples(batch_size, sequence_length=128):
    real_samples = []
    for _ in range(batch_size):
        # Choose a random MIDI file from your dataset
        random_midi_file = np.random.choice(list_of_midi_files)
        note_sequence = midi_to_note_sequence(random_midi_file, sequence_length)
        real_samples.append(note_sequence)
    return np.array(real_samples)

# Generator
def build_generator(latent_dim, sequence_length, num_features):
    input_layer = Input(shape=(latent_dim,))
    
    # Determine the number of units needed for the LSTM layer
    # to maintain the same total number of elements
    lstm_units = sequence_length * num_features
    
    x = Dense(lstm_units)(input_layer)
    x = Reshape((sequence_length, num_features))(x)
    
    # Convolutional layers
    x = Conv1DTranspose(64, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Conv1DTranspose(128, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # LSTM layer
    x = LSTM(128, return_sequences=True)(x)
    
    # Output layer
    generated_sequence = Dense(num_features, activation='tanh')(x)
    
    generator = Model(inputs=input_layer, outputs=generated_sequence)
    return generator

# Discriminator
def build_discriminator(sequence_length, num_features):
    input_layer = Input(shape=(sequence_length, num_features))
    
    x = LSTM(64, return_sequences=True)(input_layer)
    
    x = Conv1D(64, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(inputs=input_layer, outputs=x)
    return discriminator

# Combined GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    
    input_layer = Input(shape=(latent_dim,))
    generated_sequence = generator(input_layer)
    validity = discriminator(generated_sequence)
    
    gan = Model(inputs=input_layer, outputs=validity)
    return gan

# Parameters
latent_dim = 100
sequence_length = 128
num_features = 2  # using (pitch, duration) pairs as features
batch_size = 32
epochs = 100
steps_per_epoch = 100

# Build and compile the discriminator
discriminator = build_discriminator(sequence_length, num_features)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build and compile the generator
generator = build_generator(latent_dim, sequence_length, num_features)
generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Build and compile the GAN model
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training loop
for epoch in range(epochs):
    for _ in range(steps_per_epoch):
        # Generate random noise as input for the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Generate fake music sequences
        generated_sequences = generator.predict(noise)
        
        # Get a batch of real music sequences
        real_sequences = get_real_samples(batch_size)
        
        # Labels for real and fake data
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_sequences, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_sequences, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)
        
        print(f"Epoch {epoch}, Step {_}/{steps_per_epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")