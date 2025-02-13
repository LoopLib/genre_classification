# STANDARD PYTHON LIBRARIES
# Provides functions for interacting with OS (paths, files, directories)
import os
# Used to manage warning messages
import warnings

# DATA PROCESSING 
# Package for scientific computing with Python (arrays and mathematical functions)
import numpy as np 
# Used for data manipulation and analysis (DataFrame structure)
import pandas as pd

# AUDIO PROCESSING 
# Library for audio analysis, specifically for loading audio files and extracting features
import librosa

# UTILITIES
# Used to display progress bars during iterative processes
from tqdm import tqdm

###############################################################################

# Silence certain user warning from librosa to keep the console output cleaner
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

###############################################################################

def feature_extraction(df, n_mfcc=20, track_dir="data/fma_small"):
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
            try:
                y, sr = librosa.load(filename, sr=None, mono=True)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                continue
            # Skip silent or empty files
            if y is None or len(y) == 0:
                print(f"Skipping empty file: {filename}")
                continue
            
            # ============= Basic MFCC Features =============
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

            # Calculate the first-order delta of MFCCs (how MFCCS changes over time)
            delta_mfcc = librosa.feature.delta(mfcc) 
            # Compute the mean of delta MFCCs accors time for each coefficient
            delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
            # Compute the mean of delta MFCCs accors time for each coefficient
            delta_mfcc_std = np.std(delta_mfcc, axis=1)

            # Compute the root-mean-square (RMS) energy of the signal, which measures signal amplitude
            rms = librosa.feature.rms(y=y)  
            # Mean RMS value across all frames
            rms_mean = np.mean(rms)
            # Standard deviation of RMS values across all frames
            rms_std = np.std(rms)

            # ============= Spectral Features =============
            # Spectral Centroid: indicates the "center of mass" of the spectrum
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            # Mean spectral centroid across all frames
            spectral_centroid_mean = np.mean(spectral_centroid)
            # Standard deviation of the spectral centroid
            spectral_centroid_std = np.std(spectral_centroid)

            # Spectral Bandwidth: measures the spread (bandwidth) of the frequencies around the centroid
            spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            # Mean spectral bandwidth
            spectral_bw_mean = np.mean(spectral_bw)
            # Standard deviation of the spectral bandwidth
            spectral_bw_std = np.std(spectral_bw)

            # Zero Crossing Rate (ZCR): counts how often the signal crosses the zero amplitude axis
            zcr = librosa.feature.zero_crossing_rate(y)
            # Mean zero crossing rate
            zcr_mean = np.mean(zcr)
            # Standard deviation of the zero crossing rate
            zcr_std = np.std(zcr)

            mfcc_dd = librosa.feature.delta(mfcc, order=2)  # delta-delta
            mfcc_dd_mean = np.mean(mfcc_dd, axis=1)
            mfcc_dd_std = np.std(mfcc_dd, axis=1)

            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spec_rolloff_mean = np.mean(spec_rolloff)
            spec_rolloff_std = np.std(spec_rolloff)

            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spec_contrast_mean = np.mean(spec_contrast, axis=1)
            spec_contrast_std = np.std(spec_contrast, axis=1)

             # ============= Additional Features =============
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)

            # Tonnetz features computed from the harmonic component
            y_harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_std = np.std(tonnetz, axis=1)

            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y)
            flatness_mean = np.mean(flatness)
            flatness_std = np.std(flatness)

            # ============= Combine All Features Into One Row =============
            # Horizontally concatenates 1D arrays into a single 1D array
            # Reference: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
            #            https://stackoverflow.com/questions/60907414/how-to-properly-use-numpy-hstack
            feature_row = np.hstack((
                mfcc_mean, mfcc_std,
                delta_mfcc_mean, delta_mfcc_std,
                mfcc_dd_mean, mfcc_dd_std,
                [rms_mean, rms_std],
                [spectral_centroid_mean, spectral_centroid_std],
                [spectral_bw_mean, spectral_bw_std],
                [zcr_mean, zcr_std],
                [spec_rolloff_mean, spec_rolloff_std],
                spec_contrast_mean, spec_contrast_std,
                chroma_mean, chroma_std,
                tonnetz_mean, tonnetz_std,
                [flatness_mean, flatness_std]
            ))

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
