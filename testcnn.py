# STANDARD PYTHON LIBRARIES
import os
import warnings

# DATA PROCESSING 
import numpy as np 
import pandas as pd

# AUDIO PROCESSING 
import librosa

# MACHINE LEARNING 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# UTILITIES
from tqdm import tqdm
from joblib import dump, load  # For model persistence

# Import feature extraction logic from the separate file
from feature_extraction import feature_extraction

# DEEP LEARNING LIBRARIES
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

###############################################################################
# Silence certain user warning from librosa to keep the console output cleaner
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
###############################################################################

# Utility functions

def check_missing_files(df, track_dir="data/fma_small"):
    """
    Check for missing audio files based on the metadata DataFrame.
    Args:
        df (pd.DataFrame): Metadata DataFrame with a 'path' column.
        track_dir (str): Directory containing the audio files.
    Returns:
        None
    """
    missing_files = []
    for path in df["path"]:
        filepath = os.path.join(track_dir, path)
        if not os.path.isfile(filepath):
            missing_files.append(filepath)
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        for file in missing_files[:10]:  
            print(file)
    else:
        print("All files are present.")

def load_metadata_and_filter(metadata_dir="data/fma_metadata", subset="large"):
    """
    Load tracks.csv from metadata, filter rows for the chosen subset (small or large),
    and return a DataFrame containing track_id, genre, path, etc.
    """
    tracks = pd.read_csv(os.path.join(metadata_dir, "tracks.csv"), index_col=0, header=[0, 1])
    subset_mask = tracks["set", "subset"] == subset
    tracks = tracks[subset_mask]
    tracks = tracks[~tracks["track", "genre_top"].isna()]

    def track_id_to_path(tid):
        """Zero-pad TID to 6 digits, then form path 000/000123.mp3"""
        tid_str = "{:06d}".format(tid)
        return os.path.join(tid_str[:3], tid_str + ".mp3")
    
    df = pd.DataFrame(index=tracks.index)
    df["track_id"] = tracks.index
    df["genre_top"] = tracks["track", "genre_top"]
    df["subset"] = tracks["set", "subset"]
    df["path"] = df["track_id"].apply(track_id_to_path)
    return df

###############################################################################
# Main classification with a CNN model
###############################################################################

def main():
    # Define the subset of the dataset to process (small or large)
    subset = "large"
    print(f"Loading metadata for subset='{subset}'...")
    df_tracks = load_metadata_and_filter(metadata_dir="data/fma_metadata", subset=subset)
    print(f"Loaded {len(df_tracks)} tracks for subset='{subset}'")

    # Uncomment to use a smaller sample for testing:
    # df_tracks = df_tracks.head(300)

    # Remove genres with fewer than 2 samples to avoid imbalance issues
    counts = df_tracks["genre_top"].value_counts()
    valid_genres = counts[counts >= 2].index
    df_tracks = df_tracks[df_tracks["genre_top"].isin(valid_genres)]
    print("After removing classes with <2 samples, we have:")
    print(df_tracks["genre_top"].value_counts())

    df_tracks = df_tracks.reset_index(drop=True)
    df_tracks.dropna(inplace=True)

    # Check for missing audio files and remove entries for missing data
    check_missing_files(df_tracks, track_dir="data/fma_large")

    '''
    # Optionally, you can extract audio features (e.g., MFCCs) using the feature_extraction function:
    audio_dir = f"data/fma_{subset}"
    X, valid_indices = feature_extraction(df_tracks, n_mfcc=40, track_dir=audio_dir)
    df_tracks = df_tracks.iloc[valid_indices]
    y = df_tracks["genre_top"].values
    if len(X) == 0 or len(y) == 0:
        print("No valid features or labels available. Exiting...")
        return
    '''

    # Instead of extracting audio features, load precomputed features from a CSV file.
    features_path = os.path.join("data/fma_metadata", "features.csv")
    features_df = pd.read_csv(features_path, low_memory=False)
    if 'track_id' not in features_df.columns:
        features_df.rename(columns={features_df.columns[0]: 'track_id'}, inplace=True)
    features_df['track_id'] = pd.to_numeric(features_df['track_id'], errors='coerce')
    features_df = features_df.dropna(subset=['track_id'])
    features_df['track_id'] = features_df['track_id'].astype(int)

    # Merge features with metadata based on track_id.
    df_tracks = df_tracks.merge(features_df, on="track_id")
    counts_after_merge = df_tracks["genre_top"].value_counts()
    valid_genres_after_merge = counts_after_merge[counts_after_merge >= 2].index
    df_tracks = df_tracks[df_tracks["genre_top"].isin(valid_genres_after_merge)]

    # Extract feature matrix X by dropping non-feature columns.
    X = df_tracks.drop(columns=["track_id", "genre_top", "subset", "path"]).values
    y = df_tracks["genre_top"].values

    if X.shape[0] != len(y):
        print(f"Data size mismatch: X={X.shape[0]}, y={len(y)}. Exiting...")
        return

    # Encode genre labels into integers.
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    # Split data into training, validation, and test sets.
    X_train, X_temp, y_train, y_temp, df_train, df_temp = train_test_split(
        X, y_encoded, df_tracks, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test, df_val, df_test = train_test_split(
        X_temp, y_temp, df_temp, test_size=1/3, random_state=42
    )

    print("Training CNN classifier...")

    # Scale features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for 1D CNN: (samples, timesteps, channels).
    # Here, treat each feature vector as a sequence of length equal to the number of features with 1 channel.
    X_train_scaled = X_train_scaled[..., np.newaxis]
    X_val_scaled = X_val_scaled[..., np.newaxis]
    X_test_scaled = X_test_scaled[..., np.newaxis]

    # Convert labels to one-hot encoding.
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    input_shape = (X_train_scaled.shape[1], 1)

    # Build a simple CNN model.
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the CNN model.
    history = model.fit(
        X_train_scaled, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_data=(X_val_scaled, y_val_cat),
        verbose=1
    )

    # Save the trained CNN model.
    model.save("cnn_genre_classifier.h5")
    print("Trained CNN model saved to cnn_genre_classifier.h5.")

    # Evaluate on the validation set.
    val_loss, val_acc = model.evaluate(X_val_scaled, y_val_cat, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Predict on the test set.
    y_test_pred_prob = model.predict(X_test_scaled)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)

    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_enc.classes_, zero_division=0))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    predicted_labels = label_enc.inverse_transform(y_test_pred)
    actual_labels = label_enc.inverse_transform(y_test)
    df_test["predicted_genre"] = predicted_labels
    df_test["actual_genre"] = actual_labels

    with open("predictions.txt", "w") as file:
        file.write("\nSample of Test Predictions vs Actual:\n")
        for idx, row in df_test.iterrows():
            file.write(f"Path: {row['path']} | Actual: {row['actual_genre']} | Predicted: {row['predicted_genre']}\n")
    
    print("Done.")

if __name__ == "__main__":
    main()
