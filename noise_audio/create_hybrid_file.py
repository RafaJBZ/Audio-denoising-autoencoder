import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Directories containing the clean and noisy representations
clean_representations_dir = '/mnt/c/Users/rafaj/Documents/datasets/audio-denoising-auto-encoder/data/processed_data/new_metodo/train/clean_hybrid_representations'
noisy_representations_dir = '/mnt/c/Users/rafaj/Documents/datasets/audio-denoising-auto-encoder/data/processed_data/new_metodo/train/noisy_hybrid_representations'

desired_shape = (148, 109)

def load_data(directory):
    file_list = [file_name for file_name in os.listdir(directory) if file_name.endswith('.npy')]
    data = np.empty((len(file_list),) + desired_shape, dtype=np.float32)  # Use dtype=np.float32 or your desired data type
    for i, file_name in enumerate(tqdm(file_list, desc=f'Loading data from {os.path.basename(directory)}')):
        file_path = os.path.join(directory, file_name)
        representation = np.load(file_path)
        data[i] = representation
    return data

# Load clean and noisy representations concurrently
with ThreadPoolExecutor() as executor:
    clean_data_future = executor.submit(load_data, clean_representations_dir)
    noisy_data_future = executor.submit(load_data, noisy_representations_dir)

# Wait for both futures to complete
x_train = clean_data_future.result()
x_train_noisy = noisy_data_future.result()

# Print the shape of the loaded data
print("Shape of clean data:", x_train.shape)
print("Shape of noisy data:", x_train_noisy.shape)

# Save the loaded data
np.save('/mnt/c/Users/rafaj/Documents/datasets/audio-denoising-auto-encoder/data/processed_data/new_metodo/train/clean_hybrid_representations.npy', x_train)
np.save('/mnt/c/Users/rafaj/Documents/datasets/audio-denoising-auto-encoder/data/processed_data/new_metodo/train/noisy_hybrid_representations.npy', x_train_noisy)
