import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def get_available_track_ids(fma_small_dir):
    """
    Walks through the fma_small directory and collects track IDs (as integers)
    based on the file names. A progress bar is displayed using tqdm.
    """
    pattern = os.path.join(fma_small_dir, '**', '*.mp3')
    mp3_files = glob.glob(pattern, recursive=True)
    available_track_ids = set()
    
    for file in tqdm(mp3_files, desc="Scanning fma_small files"):
        try:
            # File name format: '000002.mp3'
            track_id = int(os.path.splitext(os.path.basename(file))[0])
            available_track_ids.add(track_id)
        except ValueError:
            continue
    return available_track_ids

def main():
    # Define directories
    FMA_SMALL_DIR = os.path.join("data", "fma_small")
    FMA_METADATA_DIR = os.path.join("data", "fma_metadata")
    
    # Load precomputed features with low_memory disabled.
    features_csv = os.path.join(FMA_METADATA_DIR, "features.csv")
    print("Loading features from:", features_csv)
    features_df = pd.read_csv(features_csv, index_col=0, low_memory=False)
    print(f"Features shape: {features_df.shape}")
    
    # Convert features to a numeric type (float32)
    features_df = features_df.astype('float32')
    
    # Load tracks metadata for genre labels.
    tracks_csv = os.path.join(FMA_METADATA_DIR, "tracks.csv")
    print("Loading tracks metadata from:", tracks_csv)
    # tracks.csv has a two-level header.
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    # Keep only tracks with a valid top-level genre label.
    tracks = tracks[tracks['track']['genre_top'].notnull()]
    # Create a simpler column for the genre.
    tracks['genre'] = tracks['track']['genre_top']
    
    # Get available track IDs from the fma_small directory.
    available_track_ids = get_available_track_ids(FMA_SMALL_DIR)
    print(f"\nFound {len(available_track_ids)} audio files in fma_small.")
    
    # Filter features and tracks to only include available tracks.
    features_df = features_df[features_df.index.isin(available_track_ids)]
    tracks = tracks[tracks.index.isin(available_track_ids)]
    
    # -------------------------------
    # Merge features and metadata with a progress bar
    # -------------------------------
    print("Merging features and genre metadata with progress bar...")
    # Create a dictionary for fast lookup of genre by track id.
    genre_dict = tracks['genre'].to_dict()
    genres = []
    for idx in tqdm(features_df.index, desc="Merging features"):
        genres.append(genre_dict.get(idx, None))
    # Add the genre column to features_df.
    features_df['genre'] = genres
    # Drop any rows where the genre is missing.
    data = features_df.dropna(subset=['genre'])
    print(f"Merged data has {data.shape[0]} tracks.")
    
    if data.shape[0] == 0:
        print("No tracks to process after merging features and genres. Exiting.")
        return
    
    # Separate the feature matrix and target labels.
    # Ensure that the feature matrix is of type float32.
    X = data.drop(columns=['genre']).values.astype('float32')  # Shape: (num_tracks, n_features)
    y = data['genre'].values
    
    # Encode the genre labels as integers.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Reshape X for CNN input: add a channel dimension (n_features, 1)
    X_train = X_train[..., np.newaxis]  # New shape: (num_samples, n_features, 1)
    X_test = X_test[..., np.newaxis]
    
    # Build a simple 1D CNN model.
    input_shape = X_train.shape[1:]  # (n_features, 1)
    n_classes = len(le.classes_)
    
    model = Sequential([
        # Instead of passing input_shape to a Conv1D layer, we can use an Input layer.
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # Early stopping callback to prevent overfitting.
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the CNN model.
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    
    # Evaluate the CNN model on the test set.
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
if __name__ == '__main__':
    main()
