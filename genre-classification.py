# Provides functions for interacting with OS (paths, files, directories)
import os

# Used to manage warning messages
import warnings

# Package for scientific computing with Python (arrays and mathematical functions)
import numpy as np

# Used for data manipulation and analysis (DataFrame structure)
import pandas as pd

# Library for audio analysis, specifically for loading audio files and extracting features
import librosa

# For splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# For encoding labels and scaling features
from sklearn.preprocessing import LabelEncoder, StandardScaler

# For evaluating the model
from sklearn.metrics import classification_report, confusion_matrix

# Random Forest model for classification
from sklearn.ensemble import RandomForestClassifier

# Used to display progress bars during iterative processes
from tqdm import tqdm

# For saving (dump) and loading (load) Python objects, e.g., trained models
from joblib import dump, load  # Import joblib for model persistence

###############################################################################

# Silence certain user warning from librosa to keep the console output cleaner
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

###############################################################################

# Utility functions

###############################################################################

def feature_engineering(df, n_mfcc=20, track_dir="data/fma_small"):
    """
    Extract features from audio files based on the paths given in df.
    Args:
        df (pd.DataFrame): DataFrame with at least one column 'path' pointing to mp3 files.
        n_mfcc (int): Number of MFCC features to extract.
        track_dir (str): The directory containing FMA audio (e.g., data/fma_small or data/fma_large).
    Returns:
        np.array, list: 2D array of shape (num_tracks, num_features), valid indices.
    """
    features_list = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        filename = os.path.join(track_dir, row["path"])
        try:
            # Load the audio file
            y, sr = librosa.load(filename, sr=None, mono=True)

            # Skip silent or empty files
            if y is None or len(y) == 0:
                print(f"Skipping empty file: {filename}")
                continue

            # Compute MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Concatenate mean and std to form a feature row
            feature_row = np.hstack((mfcc_mean, mfcc_std))
            features_list.append(feature_row)
            valid_indices.append(idx)

        except Exception as e:
            # Log errors and continue processing other files
            print(f"Error processing file: {filename} - {e}")
            continue

    # Convert features list to a NumPy array
    return np.array(features_list), valid_indices


