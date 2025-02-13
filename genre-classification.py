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

# MACHINE LEARNING 
# For splitting data into training and testing sets
from sklearn.model_selection import train_test_split
# For encoding labels and scaling features
from sklearn.preprocessing import LabelEncoder, StandardScaler
# For evaluating the model
from sklearn.metrics import classification_report, confusion_matrix
# Random Forest model for classification
from sklearn.ensemble import RandomForestClassifier
# Importing RandomizedSearchCV for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
# Importing pipeline
from sklearn.pipeline import Pipeline
# Import PCA to help reduce feature dimensionality
from sklearn.decomposition import PCA

# UTILITIES
# Used to display progress bars during iterative processes
from tqdm import tqdm
# For saving (dump) and loading (load) Python objects, e.g., trained models
from joblib import dump, load  # Import joblib for model persistence

# Import feature extraction logic from the separate file
from feature_extraction import feature_extraction

###############################################################################

# Silence certain user warning from librosa to keep the console output cleaner
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

###############################################################################

# Utility functions

###############################################################################

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
    check_missing_files(df_tracks, track_dir="data/fma_small")

    # Extract audio features (MFCCs) for the chosen subset
    audio_dir = f"data/fma_{subset}"
    X, valid_indices = feature_extraction(df_tracks, n_mfcc=40, track_dir=audio_dir)

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
        random_state=42, 
        stratify=y_temp
    )

    print("Optimizing RandomForestClassifier with GridSearchCV...")

    # Create a pipeline that includes scaling and classification.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.90)),  # Keep 95% of variance; can be tuned
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))
    ])

    param_grid = {
        'pca__n_components': [0.95],  # Only one value instead of 3
        'clf__n_estimators': [500],  # Reduce number of trees
        'clf__max_depth': [None, 30],  # Limit depth
        'clf__min_samples_split': [5],  # Reduce options
        'clf__min_samples_leaf': [2],  # Reduce options
        'clf__max_features': ['sqrt'],  # Single value instead of 3
        'clf__criterion': ['gini']  # Only 'gini' instead of 'entropy'
    }

    # Initialize GridSearchCV with a RandomForestClassifier
    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10,  # Try only 10 random combinations instead of all
        cv=3,  # Reduce cross-validation folds
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    # Fit grid search on the unscaled training data (scaling is done inside the pipeline).
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)

    # Use the best estimator from grid search.
    clf = grid_search.best_estimator_

    # Save the trained model
    model_filename = "random_forest_genre_classifier.joblib"
    dump(clf, model_filename)
    print(f"Trained model saved to {model_filename}.")

    # Evaluate on the validation set
    print("\nEvaluating on the validation set...")

    y_val_pred = clf.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=label_enc.classes_, zero_division=0))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    y_test_pred = clf.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_enc.classes_, zero_division=0))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # === INVERSE TRANSFORM to get string labels ===
    predicted_labels = label_enc.inverse_transform(y_test_pred)
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