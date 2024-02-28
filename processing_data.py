import os
import numpy as np
import librosa.feature
import librosa.display
import librosa
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import joblib
from tqdm import tqdm
import shutil


class AudioProcessor:
    """
    Class to process audio files and compute Mel spectrogram and MFCC features.
    """

    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20, fixed_length=55296):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.fixed_length = fixed_length
        self.scal = StandardScaler()

    def compute_mel_mfcc(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Mel spectrogram and MFCC features for a given audio file.

        Parameters:
            audio_path (str): Path to the audio file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing Mel spectrogram and MFCC features.
        """
        y, _ = librosa.load(audio_path, sr=self.sr)
        y = librosa.util.fix_length(y, size=self.fixed_length)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                                             n_mels=self.n_mels)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=self.n_mfcc)
        mfcc = self.scal.fit_transform(mfcc)
        return mel, mfcc

    @staticmethod
    def add_noise(mel: np.ndarray, mfcc: np.ndarray, mean=0, std=0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add Gaussian noise to Mel spectrogram and MFCC features.

        Parameters:
            mel (np.ndarray): Mel spectrogram.
            mfcc (np.ndarray): MFCC features.
            mean (float): Mean of the Gaussian noise. Default is 0.
            std (float): Standard deviation of the Gaussian noise. Default is 0.05.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing noisy Mel spectrogram and MFCC features.
        """
        mel_noisy = mel + np.random.normal(mean, std, mel.shape)
        mfcc_noisy = mfcc + np.random.normal(mean, std, mfcc.shape)
        return mel_noisy, mfcc_noisy


class RepresentationSaver:
    """
    Class to save hybrid representations to files.
    """

    def __init__(self):
        pass

    @staticmethod
    def save_hybrid_representations(audio_paths: list, clean_save_dir: str, noisy_save_dir: str,
                                    processor: AudioProcessor):
        """
        Save hybrid representations (clean and noisy) to files.

        Parameters:
            audio_paths (list): List of paths to audio files.
            clean_save_dir (str): Directory to save clean representations.
            noisy_save_dir (str): Directory to save noisy representations.
            processor (AudioProcessor): Instance of AudioProcessor class.
        """
        for audio_path in tqdm(audio_paths, desc='Processing audio files'):
            mel_clean, mfcc_clean = processor.compute_mel_mfcc(audio_path)
            mel_noisy, mfcc_noisy = processor.add_noise(mel_clean, mfcc_clean)

            filename = os.path.splitext(os.path.basename(audio_path))[0]

            np.save(os.path.join(clean_save_dir, f'{filename}_hybrid_representation_clean.npy'),
                    np.concatenate((mel_clean, mfcc_clean), axis=0))

            np.save(os.path.join(noisy_save_dir, f'{filename}_hybrid_representation_noisy.npy'),
                    np.concatenate((mel_noisy, mfcc_noisy), axis=0))


def main():
    # Paths
    audio_dir = '/mnt/c/Users/rafaj/Documents/datasets/audio-denoising-auto-encoder/data/flickr_audio/wavs'
    train_clean_representations_dir = '/mnt/c/Users/rafaj/Documents/datasets/audio-denoising-auto-encoder/data/processed_data/train/clean_hybrid_representations'
    train_noisy_representations_dir = '/mnt/c/Users/rafaj/Documents/datasets/audio-denoising-auto-encoder/data/processed_data/train/noisy_hybrid_representations'
    test_audio_dir = '/mnt/c/Users/rafaj/Documents/datasets/audio-denoising-auto-encoder/data/processed_data/test'

    # Create directories if they don't exist
    os.makedirs(train_clean_representations_dir, exist_ok=True)
    os.makedirs(train_noisy_representations_dir, exist_ok=True)
    os.makedirs(test_audio_dir, exist_ok=True)

    processor = AudioProcessor()
    saver = RepresentationSaver()

    # Split audio files into train and test sets
    audio_paths = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith('.wav')]
    np.random.shuffle(audio_paths)  # Shuffle the list of audio paths
    num_train = int(len(audio_paths) * 0.9)  # 90% for training, 10% for testing
    train_paths = audio_paths[:num_train]
    test_paths = audio_paths[num_train:]

    # Process and save hybrid representations for training set
    saver.save_hybrid_representations(train_paths, train_clean_representations_dir, train_noisy_representations_dir,
                                      processor)
    print('Hybrid representations for training set saved successfully!')

    # Copy test audio files to test directory
    for path in test_paths:
        filename = os.path.basename(path)
        dest_path = os.path.join(test_audio_dir, filename)
        shutil.copyfile(path, dest_path)
    print('Test audio files copied successfully!')

    # Save scaler
    joblib.dump(processor.scal,
                '/mnt/c/Users/rafaj/Documents/datasets/audio-denoising-auto-encoder/data/processed_data/weights/scaler.save')
    print('Scaler saved successfully!')


if __name__ == "__main__":
    main()

