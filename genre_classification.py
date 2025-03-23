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

# MACHINE LEARNING 
# For splitting data into training and testing sets
from sklearn.model_selection import train_test_split
# For encoding labels and scaling features
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import feature extraction logic from the separate file
from feature_extraction import feature_extraction

# Import shared utility functions
from genre_classification.data_utils import check_missing_files, load_metadata_and_filter

# Import model training functions
from models.rf import train_rf_model
from genre_classification.models.cnn_features import train_cnn_model

# Silence certain user warning from librosa to keep the console output cleaner
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


def main():

    # Define the subset of the dataet to process (this project uses small and large)
    subset = "small"

    print(f"Loading metadata for subset='{subset}'...")

    # Load metadata and filter for the chosen subset
    df_tracks = load_metadata_and_filter(metadata_dir="data/fma_metadata", subset=subset)
    print(f"Loaded {len(df_tracks)} tracks for subset='{subset}'")

    # Only first X tracks for faster testing
    # df_tracks = df_tracks.head(300)

    # Remove genres with fewer than 2 samples to avoid imbalance issues
    counts = df_tracks["genre_top"].value_counts()
    valid_genres = counts[counts >= 2].index
    df_tracks = df_tracks[df_tracks["genre_top"].isin(valid_genres)]
    print("After removing classes with <2 samples, we have:", df_tracks["genre_top"].value_counts())

    # Reset the index of DataFrame to ensure alignment with valid_indicies later
    df_tracks = df_tracks.reset_index(drop=True)

    # Remove rows with missing values before filtering based on valid indices
    df_tracks.dropna(inplace=True)

    # Check for missing audio files and remove entries for missing data
    check_missing_files(df_tracks, track_dir="data/fma_large")

    ''' # Extract audio features (MFCCs) for the chosen subset
    audio_dir = f"data/fma_{subset}"
    X, valid_indices = feature_extraction(df_tracks, n_mfcc=40, track_dir=audio_dir)

    # Use valid_indices to filter the original DataFrame to include only valid data
    df_tracks = df_tracks.iloc[valid_indices]
    y = df_tracks["genre_top"].values

    # Handle cases where no features or lables are extracted
    if len(X) == 0 or len(y) == 0:
        print("No valid features or labels available. Exiting...")
        return
    ''' 

    # Instead of extracting audio features (MFCCs) for the chosen subset using feature_extraction,
    # load precomputed features from CSV file.
    features_path = os.path.join("data/fma_metadata", "features.csv")
    # Load features.csv with low_memory=False and treat the first column as the index.
    # Then reset the index and rename it to "track_id" for merging.
    features_df = pd.read_csv(features_path, low_memory=False)
    if 'track_id' not in features_df.columns:
        features_df.rename(columns={features_df.columns[0]: 'track_id'}, inplace=True)
    # Convert track_id to numeric, coercing errors to NaN, then drop non-numeric rows
    features_df['track_id'] = pd.to_numeric(features_df['track_id'], errors='coerce')
    features_df = features_df.dropna(subset=['track_id'])
    features_df['track_id'] = features_df['track_id'].astype(int)


    # Merge the features with the metadata DataFrame based on track_id
    df_tracks = df_tracks.merge(features_df, on="track_id")

    # Filter out genres that now have fewer than 2 samples after the merge
    counts_after_merge = df_tracks["genre_top"].value_counts()
    valid_genres_after_merge = counts_after_merge[counts_after_merge >= 2].index
    df_tracks = df_tracks[df_tracks["genre_top"].isin(valid_genres_after_merge)]

    # Extract feature matrix X by dropping non-feature columns
    # Assuming features.csv contains feature columns along with 'track_id'
    X = df_tracks.drop(columns=["track_id", "genre_top", "subset", "path"]).values
    y = df_tracks["genre_top"].values

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
    X_train, X_temp, y_train, y_temp, df_train, df_temp = train_test_split(
        X,                      # Features
        y_encoded,              # Labels
        df_tracks,              # DataFrame with metadata
        test_size=0.3,          # 30% of the data is used for rest
        random_state=42,        # Set a seed for random number generation
        stratify=y_encoded      # Ensures that the distribution of classes is 
                                # similar in both training and test sets
    )

    X_val, X_test, y_val, y_test, df_val, df_test = train_test_split(
        X_temp,
        y_temp, 
        df_temp, 
        test_size=1/3, 
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Model Selection ---

    # To train the Random Forest model:
    # train_rf_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, label_enc, df_test, scaler)

    # To train the CNN model:
    # Reshape for CNN (add channel dimension)
    X_train_cnn = X_train_scaled[..., np.newaxis]
    X_val_cnn = X_val_scaled[..., np.newaxis]
    X_test_cnn = X_test_scaled[..., np.newaxis]
    
    train_cnn_model(X_train_cnn, y_train, X_val_cnn, y_val, X_test_cnn, y_test, label_enc, df_test)

if __name__ == "__main__":
    main()