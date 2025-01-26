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

    # Initialize an empty list to store extracted feature rows for each track
    features_list = []

    # Initialize an empty list to store indicies of successfully processed tracks
    valid_indices = []

    # Iterate over each row in the DataFrame with a progress bar
    # Reference: https://stackoverflow.com/questions/47087741/use-tqdm-progress-bar-with-pandas
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        # Construct the full file path by combining the directory and file path in the DataFrame
        filename = os.path.join(track_dir, row["path"])

        try:
            # Load the audio file with librosa
            # Sample rate (sr)
                # The sr parameter specifies the desired sampling rate for the loaded audio
                # If sr=None, the audio file is loaded with the native sampling rate, 
                # the rate it was originally stored in
            # Mono
                # If mono = true, the audio is converted to a single channel by
                # averaging all the channels, it helps to simplify analysis-
                # removes stereo and multi-channel information
            # Mono is set to True to convert the audio to mono (average channels)
            # Reference: https://librosa.org/doc/main/generated/librosa.load.html
            y, sr = librosa.load(filename, sr=None, mono=True)

            # Skip silent or empty files
            if y is None or len(y) == 0:
                print(f"Skipping empty file: {filename}")
                continue
            
            # MEAN AND STANDARD DEVIATION OF MFCCS
            # Compute Mel-frequency cepstral coefficients (MFCCs) from the audio
            # Input Audio Signal (y) 
                # The raw audio waveform (as a 1D NumPy array) that the function will 
                # process to extract features
            # Sampling Rate (sr)
                # The sampling rate of the audio signal (in Hz)
                # Example: a 44,100 Hz sampling rate = 22,050 Nyquist frequency
            # The number of MFCCS to compute (n_mfcc)
                # The number of MFCC coefficient to calculate for each time frame of the audio file
            # The MFCCS are returned as a 2D array where:
                # Rows = Number of MFCCs (n_mfcc)
                # Columns = Number of frames in the audio file
            # Reference: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            # This operation computes the time varying MFCC features into a single representative 
            # value (mean) for each coefficient
            # Output is a 1D array of shape (n_mfcc,)
            mfcc_mean = np.mean(mfcc, axis=1)

            # Computes standard deviation of the Mel-frequncy cepstral coefficients accross time
            # Used to quantify how much variation exists in each MFCC coefficient over duration
            # of the audio signal
            mfcc_std = np.std(mfcc, axis=1)

            # Horizontally concatenates two 1D arrays: mfcc_mean and mfcc_std
            # Reference: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
            #            https://stackoverflow.com/questions/60907414/how-to-properly-use-numpy-hstack
            
            feature_row = np.hstack((mfcc_mean, mfcc_std))

            # Append the feature row to the list of extracted features
                # Each row corresponds to a single track
                # Each row contains the extracted features for that track
            features_list.append(feature_row)

            # Append the index of the valid track to the list of valid indices
            # idx
                # The index of the current row in the DataFrame
                # Obtained from the iterrows() method
            valid_indices.append(idx)

        except Exception as e:
            # Log errors and continue processing other files
            print(f"Error processing file: {filename} - {e}")
            continue

    # Convert features list to a 2D NumPy array
    return np.array(features_list), valid_indices


