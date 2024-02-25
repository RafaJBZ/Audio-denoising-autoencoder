import os
import numpy as np
import librosa.feature
import librosa.display
import librosa
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import joblib
from tqdm import tqdm

# Parameters for audio feature extraction
hop_length = 512
n_mels = 128
n_mfcc = 20
n_fft = 2048
sr = 22050
fixed_length = 55296

# Initialize StandardScaler for MFCC scaling
scal = StandardScaler()


def compute_mel_mfcc(audio_path: str, sr: int, n_fft: int, hop_length: int, n_mels: int, n_mfcc: int,
                     fixed_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Mel spectrogram and MFCC features for an audio file.

    Parameters:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate of the audio.
        n_fft (int): Length of the FFT window.
        hop_length (int): Hop length for the STFT.
        n_mels (int): Number of Mel bins.
        n_mfcc (int): Number of MFCC coefficients to compute.
        fixed_length (int): Desired fixed length of the audio.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing Mel spectrogram and MFCC features.
    """
    # Load audio file
    y, _ = librosa.load(audio_path)

    # Ensure fixed length
    y = librosa.util.fix_length(y, size=fixed_length)

    # Compute Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    # Scale MFCC
    mfcc = scal.fit_transform(mfcc)

    return mel, mfcc


def add_noise(mel: np.ndarray, mfcc: np.ndarray, mean=0, std=0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise to Mel spectrogram and MFCC features.

    Parameters:
        mel (np.ndarray): Mel spectrogram features.
        mfcc (np.ndarray): MFCC features.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing noisy Mel spectrogram and MFCC features.
    """
    # Add noise to Mel spectrogram and MFCC
    mel_noisy = mel + np.random.normal(mean, std, mel.shape)
    mfcc_noisy = mfcc + np.random.normal(mean, std, mfcc.shape)

    return mel_noisy, mfcc_noisy


def save_hybrid_representations(audio_paths: list, clean_save_dir: str, noisy_save_dir: str) -> None:
    """
    Process audio files, add noise, and save hybrid representations.

    Parameters:
        audio_paths (list): List of paths to audio files.
        clean_save_dir (str): Directory to save clean hybrid representations.
        noisy_save_dir (str): Directory to save noisy hybrid representations.

    Returns:
        None
    """
    for audio_path in tqdm(audio_paths, desc='Processing audio files'):
        # Compute Mel spectrogram and MFCC for clean audio
        mel_clean, mfcc_clean = compute_mel_mfcc(audio_path, sr, n_fft, hop_length, n_mels, n_mfcc, fixed_length)

        # Compute Mel spectrogram and MFCC for noisy audio
        mel_noisy, mfcc_noisy = add_noise(mel_clean, mfcc_clean)

        # Save clean representations
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        np.save(os.path.join(clean_save_dir, f'{filename}_hybrid_representation_clean.npy'),
                np.concatenate((mel_clean, mfcc_clean), axis=0))

        # Save noisy representations
        np.save(os.path.join(noisy_save_dir, f'{filename}_hybrid_representation_noisy.npy'),
                np.concatenate((mel_noisy, mfcc_noisy), axis=0))


def main():
    # Paths
    audio_dir = 'data/flickr_audio/wavs'
    clean_representations_dir = 'data/processed_data/clean_hybrid_representations'
    noisy_representations_dir = 'data/processed_data/noisy_hybrid_representations'

    # Create directories if they don't exist
    os.makedirs(clean_representations_dir, exist_ok=True)
    os.makedirs(noisy_representations_dir, exist_ok=True)

    # Process audio files and save hybrid representations
    audio_paths = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith('.wav')]
    save_hybrid_representations(audio_paths, clean_representations_dir, noisy_representations_dir)

    # Save scaler
    joblib.dump(scal, 'data/processed_data/weights/scaler.save')

    print('Hybrid representations saved successfully!')


if __name__ == "__main__":
    main()
