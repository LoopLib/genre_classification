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
from joblib import dump, load  # Still available if needed for other purposes

# DEEP LEARNING (CNN)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

###############################################################################
# Silence certain user warnings from librosa to keep the console output cleaner
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

###############################################################################
# Utility functions

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
    features_list = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        filename = os.path.join(track_dir, row["path"])
        try:
            try:
                y, sr = librosa.load(filename, sr=None, mono=True)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                continue

            if y is None or len(y) == 0:
                print(f"Skipping empty file: {filename}")
                continue

            # ============= Basic MFCC Features =============
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            delta_mfcc = librosa.feature.delta(mfcc) 
            delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
            delta_mfcc_std = np.std(delta_mfcc, axis=1)

            rms = librosa.feature.rms(y=y)  
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)

            # ============= Spectral Features =============
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroid)
            spectral_centroid_std = np.std(spectral_centroid)

            spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_bw_mean = np.mean(spectral_bw)
            spectral_bw_std = np.std(spectral_bw)

            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)

            # ============= Combine All Features Into One Row =============
            feature_row = np.hstack((
                mfcc_mean, mfcc_std,
                delta_mfcc_mean, delta_mfcc_std,
                [rms_mean, rms_std],
                [spectral_centroid_mean, spectral_centroid_std],
                [spectral_bw_mean, spectral_bw_std],
                [zcr_mean, zcr_std],
            ))

            features_list.append(feature_row)
            valid_indices.append(idx)

        except Exception as e:
            print(f"Error processing file: {filename} - {e}")
            continue

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

def load_metadata_and_filter(metadata_dir="data/fma_metadata", subset="small"):
    """
    Load tracks.csv from metadata, filter rows for the chosen subset (small or large),
    and return a DataFrame containing track_id, genre, path, etc.
    """
    tracks = pd.read_csv(os.path.join(metadata_dir, "tracks.csv"), index_col=0, header=[0, 1])
    subset_mask = tracks["set", "subset"] == subset
    tracks = tracks[subset_mask]
    tracks = tracks[~tracks["track", "genre_top"].isna()]

    def track_id_to_path(tid):
        tid_str = "{:06d}".format(tid)
        return os.path.join(tid_str[:3], tid_str + ".mp3")
    
    df = pd.DataFrame(index=tracks.index)
    df["track_id"] = tracks.index
    df["genre_top"] = tracks["track", "genre_top"]
    df["subset"] = tracks["set", "subset"]
    df["path"] = df["track_id"].apply(track_id_to_path)
    return df

###############################################################################
# Main classification using a CNN model

def main():

    subset = "small"
    print(f"Loading metadata for subset='{subset}'...")
    df_tracks = load_metadata_and_filter(metadata_dir="data/fma_metadata", subset=subset)
    print(f"Loaded {len(df_tracks)} tracks for subset='{subset}'")

    # (Optional) For faster testing, use only the first 300 tracks.
    # df_tracks = df_tracks.head(300)

    # Remove genres with fewer than 2 samples.
    counts = df_tracks["genre_top"].value_counts()
    valid_genres = counts[counts >= 2].index
    df_tracks = df_tracks[df_tracks["genre_top"].isin(valid_genres)]
    print("After removing classes with <2 samples, we have:")
    print(df_tracks["genre_top"].value_counts())

    df_tracks = df_tracks.reset_index(drop=True)
    df_tracks.dropna(inplace=True)

    # Check for missing audio files.
    check_missing_files(df_tracks, track_dir="data/fma_small")

    # Extract audio features.
    audio_dir = f"data/fma_{subset}"
    X, valid_indices = feature_extraction(df_tracks, n_mfcc=20, track_dir=audio_dir)
    df_tracks = df_tracks.iloc[valid_indices]
    y = df_tracks["genre_top"].values

    if len(X) == 0 or len(y) == 0:
        print("No valid features or labels available. Exiting...")
        return

    if X.shape[0] != len(y):
        print(f"Data size mismatch: X={X.shape[0]}, y={len(y)}. Exiting...")
        return

    # Encode genre labels.
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    # Split the data into training, validation, and test sets.
    X_train, X_temp, y_train, y_temp, df_train, df_temp = train_test_split(
        X, y_encoded, df_tracks, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test, df_val, df_test = train_test_split(
        X_temp, y_temp, df_temp, test_size=1/3, random_state=42, stratify=y_temp
    )

    # Standardize the features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the data for the CNN: (samples, n_features, channels)
    # Here, n_features is the length of the feature vector and we use 1 channel.
    X_train_scaled = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
    X_val_scaled = X_val_scaled.reshape(-1, X_val_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

    # Convert labels to categorical (one-hot encoding)
    num_classes = len(np.unique(y_encoded))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    print("Building and training the CNN model...")

    # Build the CNN model with Batch Normalization.
    model = Sequential([
        # First Conv Block
        Conv1D(filters=64, kernel_size=3, padding='same', input_shape=(X_train_scaled.shape[1], 1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        
        # Second Conv Block
        Conv1D(filters=128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        
        # Fully Connected Layers
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define callbacks to prevent overfitting and reduce learning rate on plateau.
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Train the model.
    history = model.fit(
        X_train_scaled, y_train_cat,
        epochs=30,
        batch_size=16,
        validation_data=(X_val_scaled, y_val_cat),
        callbacks=[early_stop, lr_reduce],
        verbose=1
    )

    # Save the trained model.
    model_filename = "cnn_genre_classifier.h5"
    model.save(model_filename)
    print(f"Trained CNN model saved to {model_filename}.")

    # Evaluate on the validation set.
    print("\nEvaluating on the validation set...")
    val_loss, val_acc = model.evaluate(X_val_scaled, y_val_cat, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    y_val_pred_prob = model.predict(X_val_scaled)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=label_enc.classes_, zero_division=0))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    # Evaluate on the test set.
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    y_test_pred_prob = model.predict(X_test_scaled)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_enc.classes_, zero_division=0))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # Inverse transform to get string labels.
    predicted_labels = label_enc.inverse_transform(y_test_pred)
    actual_labels = label_enc.inverse_transform(y_test)

    # Store predictions in df_test.
    df_test["predicted_genre"] = predicted_labels
    df_test["actual_genre"] = actual_labels

    # Write predictions to file.
    with open("predictions.txt", "w") as file:
        file.write("\nSample of Test Predictions vs Actual:\n")
        for idx, row in df_test.iterrows():
            file.write(f"Path: {row['path']} | Actual: {row['actual_genre']} | Predicted: {row['predicted_genre']}\n")
    
    print("Done.")

    # Plot training & validation accuracy and loss values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
