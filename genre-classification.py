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

            # ============= Combine All Features Into One Row =============
            # Horizontally concatenates 1D arrays into a single 1D array
            # Reference: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
            #            https://stackoverflow.com/questions/60907414/how-to-properly-use-numpy-hstack
            feature_row = np.hstack((
            mfcc_mean, mfcc_std,
            delta_mfcc_mean, delta_mfcc_std,
            mfcc_dd_mean, mfcc_dd_std,
            rms_mean, rms_std,
            spectral_centroid_mean, spectral_centroid_std,
            spectral_bw_mean, spectral_bw_std,
            zcr_mean, zcr_std,
            spec_rolloff_mean, spec_rolloff_std,
            spec_contrast_mean, spec_contrast_std
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

def check_missing_files(df, track_dir="data/fma_small"):
    """
    Check for missing audio files based on the metadata DataFrame.
    Args:
        df (pd.DataFrame): Metadata DataFrame with a 'path' column.
        track_dir (str): Directory containing the audio files.
    Returns:
        None
    """
    # Initialize an empty list to keep track of missing files
    missing_files = []

    # Iterate over each path in the path column of the DataFrame
    for path in df["path"]:

        # Construct the full file path by combining the directory and file path
        filepath = os.path.join(track_dir, path)

        # Check if the file does not exist at the specified path
        if not os.path.isfile(filepath):
            # If file is missing add to the list of missing files
            missing_files.append(filepath)

    # Check if there is any missing files
    if missing_files:
        # Print the total count of missing files
        print(f"Missing files: {len(missing_files)}")

        # Display the first 10 missing files for review
        for file in missing_files[:10]:  
            print(file)
    else:
        # If no missing files, print confirmation message
        print("All files are present.")

def load_metadata_and_filter(metadata_dir="data/fma_metadata", subset="small"):
    """
    Load tracks.csv from metadata, filter rows for the chosen subset (small or large),
    and return a DataFrame containing track_id, genre, path, etc.
    """
    # Load the tracks.csv file into DataFrame
    # Specify first column as the index and use multi-level header
    tracks = pd.read_csv(os.path.join(metadata_dir, "tracks.csv"), index_col=0, header=[0, 1])

    # Filter the DataFrame rows to include only those belonging to the specified
    # subset (small or large)
    subset_mask = tracks["set", "subset"] == subset
    tracks = tracks[subset_mask]

    # Remove rows where the genre_top is NaN (missing genre)
    tracks = tracks[~tracks["track", "genre_top"].isna()]

    # Function to generate file path for a track_id
    def track_id_to_path(tid):
        """Zero-pad TID to 6 digits, then form path 000/000123.mp3"""
        # Convert track ID to a zero-padded string of 6 digits.
        tid_str = "{:06d}".format(tid)
        # Concatenate the first 3 digits as a folder and add the track ID as a file name.
        return os.path.join(tid_str[:3], tid_str + ".mp3")
    
    # Create a new DataFrame to store the filtered data.
    df = pd.DataFrame(index=tracks.index)

   # Add track_id as a column.
    df["track_id"] = tracks.index
   
    # Add genre_top column from the tracks DataFrame.
    df["genre_top"] = tracks["track", "genre_top"]

    # Add subset column from the tracks DataFrame.
    df["subset"] = tracks["set", "subset"]

    # Generate the path for each track_id using the helper function.
    df["path"] = df["track_id"].apply(track_id_to_path)
 
    # Return the resulting filtered DataFrame.
    return df

###############################################################################

# Main classification

###############################################################################

def main():

    # Define the subset of the dataet to process (this project uses small and large)
    subset = "small"

    print(f"Loading metadata for subset='{subset}'...")

    # Load metadata and filter for the chosen subset
    df_tracks = load_metadata_and_filter(metadata_dir="data/fma_metadata", subset=subset)
    print(f"Loaded {len(df_tracks)} tracks for subset='{subset}'")

    # Only first X tracks for faster testing
    df_tracks = df_tracks.head(100)

    # Remove genres with fewer than 2 samples to avoid imbalance issues
    counts = df_tracks["genre_top"].value_counts()
    valid_genres = counts[counts >= 2].index
    df_tracks = df_tracks[df_tracks["genre_top"].isin(valid_genres)]
    print("After removing classes with <2 samples, we have:", df_tracks["genre_top"].value_counts())

    # Reset the index of DataFrame to ensure alignment with valid_indicies later
    df_tracks = df_tracks.reset_index(drop=True)

    # Check for missing audio files and remove entries for missing data
    check_missing_files(df_tracks, track_dir="data/fma_small")

    # Extract audio features (MFCCs) for the chosen subset
    audio_dir = f"data/fma_{subset}"
    X, valid_indices = feature_engineering(df_tracks, n_mfcc=20, track_dir=audio_dir)

    # Remove rows with missing values before filtering based on valid indices
    df_tracks.dropna(inplace=True)

    # Use valid_indices to filter the original DataFrame to include only valid data
    df_tracks = df_tracks.iloc[valid_indices]
    y = df_tracks["genre_top"].values

    # Handle cases where no features or lables are extracted
    if len(X) == 0 or len(y) == 0:
        print("No valid features or labels available. Exiting...")
        return

    # Ensure consistent sizes for feature matrix X and label vector y
    if X.shape[0] != len(y):
        print(f"Data size mismatch: X={X.shape[0]}, y={len(y)}. Exiting...")
        return

    # Encode genre labels into integers for classification 
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    # Split the data into training and testing sets
        # Training set: 80% of the data, used to train the model
        # Test set: 20% of the data, used to evaluate the model's performance
    # Reference: https://realpython.com/train-test-split-python-data/
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,                      # Features
        y_encoded,              # Labels
        df_tracks,              # DataFrame with metadata
        test_size=0.2,          # 20% of the data is used for testing
        random_state=42,        # Set a seed for random number generation
        stratify=y_encoded      # Ensures that the distribution of classes is 
                                # similar in both training and test sets
    )

    # Initializxe the scaler to standarize the features in dataset
    # Reference: https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()

    # fit_transform
        # fit - Computes the mean and standard deviation of each feature from the training data
        # transform - Scales the training data using the computed mean and standard deviation
    # The result contains the scaled versiom of the training data (X_train_scaled)
    X_train_scaled = scaler.fit_transform(X_train)

    # transform - Uses the same mean and standard deviation computed from the training data
    # to scale the test data (X_test)
    # This ensures consistency between training and test data
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier...")

    # Create an instance of a Random Forest classifier
    # Uses mutiple decision trees to perform classification tasks
    # n_estimators
        # The number of trees in the forest
        # The higher the number of trees, the better the model performance
    # random_state
        # Sets a seed for random number generatio to ensure reproducibility of results
        # Ensures consistent results when the code is run mutiple times
    # n_jobs
        # Allowing classifier to use all avaiavle CPU cores for parallel computation,
        # speeding up the training process
    # class_weight = "balanced"
        # Adjusts the weights of the classes to balance the dataset
        # Helps to improve the model's performance on imbalanced datasets
    # Reference: https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")

    # Fits the Random Forest model using provided training data
    # X_train_scaled
        # Represents the scaled features of the training dataset
    # y_train
        # Represents the target labels for the training data.
    clf.fit(X_train_scaled, y_train)

    # Save the trained model
    model_filename = "random_forest_genre_classifier.joblib"
    dump(clf, model_filename)
    print(f"Trained model saved to {model_filename}.")

    print("Evaluating...")

    # Genre predictions on the test set
    y_pred = clf.predict(X_test_scaled)

    # Print classification metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

    # Print confusion matric to evaluate the classifier performance
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # === INVERSE TRANSFORM to get string labels ===
    predicted_labels = label_enc.inverse_transform(y_pred)
    actual_labels = label_enc.inverse_transform(y_test)

    # === Store in df_test ===
    df_test["predicted_genre"] = predicted_labels
    df_test["actual_genre"] = actual_labels

   # === Display each track's actual vs. predicted genre ===
    with open("predictions.txt", "w") as file:
        file.write("\nSample of Test Predictions vs Actual:\n")
        for idx, row in df_test.iterrows():
            file.write(f"Path: {row['path']} | Actual: {row['actual_genre']} | Predicted: {row['predicted_genre']}\n")
        print("Done.")

if __name__ == "__main__":
    main()
