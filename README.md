# Sonic Saviors: Autoencoders for Audio Noise Reduction

## Project Overview

Welcome to Sonic Saviors, where we delve into the realm of noise reduction in audio signals using the power of autoencoders. This project explores two innovative approaches:

- **Feature-Level Denoising:** We extract intricate features like MFCC and Mel Spectrogram from pristine audio, introduce noise, and train an autoencoder to reconstruct these features, effectively cleansing the signal of unwanted noise.

- **Audio-Level Denoising:** In this approach, we directly manipulate the raw audio waveform by adding noise and subsequently extracting MFCC and Mel Spectrogram features. The autoencoder then works its magic to restore the clean audio waveform, employing these features.

## Project Structure

- **README.md:** Your compass through this auditory adventure.
- **data/:** Dive into this directory to uncover the audio data used for our experiments.
- **flickr_audio_eda.ipynb:** Embark on an auditory journey with this Jupyter Notebook containing exploratory data analysis (EDA) for audio data (if applicable).
- **noise_audio/:** Here resides the arsenal of scripts dedicated to audio-level denoising.
  - **create_hybrid_file.py:** Craft hybrid files melding clean and noisy audio representations.
  - **denoisening_dae_audio.ipynb:** Unravel the mysteries of audio-level denoising with this Jupyter Notebook.
  - **processing_data.py:** Script to preprocesse audio data for audio-level denoising (if needed).
- **noise_features/:** This directory houses scripts related to feature-level denoising.
  - **create_hybrid_file.py:** Generates hybrid files combining clean and noisy feature representations.
  - **denoisening_dae_features.ipynb:** Explore feature-level denoising techniques with this Jupyter Notebook.
  - **processing_data.py:** Script to preprocess audio data for feature-level denoising (if needed).
- **prediction.ipynb:** Witness the prowess of our trained autoencoder models in action through this Jupyter Notebook.

## Autoencoder Architecture

Behold the architecture defined within the `create_autoencoder` function, serving as the backbone for both models.

```python
def create_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)  # (148, 109, 1)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)

    # Latent space
    latent_space = Dense(128, activation='relu')(x)

    # Decoder
    x = Dense(37 * 28 * 64, activation='relu')(latent_space)
    x = Reshape((37, 28, 64))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Cropping2D(cropping=((0, 0), (0, 1)))(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Cropping2D(cropping=((0, 0), (1, 0)))(x)
    output_layer = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder

```
## Preprocessing Parameters

- **Sampling rate (sr):** 22050 Hz
- **Fast Fourier Transform (FFT) window size (n_fft):** 2048
- **Hop length (hop_length):** 512
- **Number of Mel-frequency cepstral coefficients (n_mels):** 128
- **Number of MFCC coefficients (n_mfcc):** 20
- **Fixed length for feature vectors (fixed_length):** 55296
