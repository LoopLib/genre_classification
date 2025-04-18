import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
import numpy as np

# -------------------------------------------------------------------
# 1) CONFIG & PATHS
# -------------------------------------------------------------------
TRACKS_CSV = "data/fma_metadata/tracks.csv"
SPECTROGRAM_DIR = "data/fma_small_spectrograms"  # subfolders: 000, 001, ..., 155
IMG_SIZE = (128, 128)   # Resize dimension
BATCH_SIZE = 32
EPOCHS = 15  

# If you want to limit how many subfolders to process for a quick test:
MAX_FOLDERS = None 

# -------------------------------------------------------------------
# 2) LOAD TRACKS.CSV & GET TOP-LEVEL GENRE
# -------------------------------------------------------------------
print("[INFO] Loading tracks.csv...")
tracks_df = pd.read_csv(TRACKS_CSV, header=[0,1], index_col=0)
df_track = tracks_df['track']  # Sub-DataFrame with columns like 'genre_top'

# Drop tracks that have no top-level genre
df_track = df_track.dropna(subset=['genre_top'])

# Dictionary: track_id -> top-level genre (string)
track_to_topgenre = df_track['genre_top'].to_dict()

# Gather all unique top-level genre names
all_genres = sorted(list(df_track['genre_top'].unique()))
genre_to_idx = {g: i for i, g in enumerate(all_genres)}
idx_to_genre = {i: g for g, i in genre_to_idx.items()}

print("[INFO] Found top-level genres:", all_genres)

# -------------------------------------------------------------------
# 3) LOAD SPECTROGRAM IMAGES & ASSIGN LABELS (WITH DEBUG PRINTS)
# -------------------------------------------------------------------
print("[INFO] Scanning spectrogram folders...")
folders = sorted(os.listdir(SPECTROGRAM_DIR))
if MAX_FOLDERS is not None:
    folders = folders[:MAX_FOLDERS]  # limit for quick debugging

images = []
labels = []
num_files_total = 0

t0 = time.time()
for folder_i, folder in enumerate(folders):
    folder_path = os.path.join(SPECTROGRAM_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"[DEBUG] Processing folder {folder_i+1}/{len(folders)}: '{folder}'")
    file_list = os.listdir(folder_path)

    for file_i, fname in enumerate(file_list):
        if fname.endswith(".png"):
            # Parse track ID (e.g. "000002.png" -> track_id=2)
            track_str = fname.replace(".png", "")
            track_id = int(track_str)

            # If the track isn't in our dictionary or the genre is NaN, skip
            if track_id not in track_to_topgenre:
                continue
            genre_name = track_to_topgenre[track_id]
            if pd.isna(genre_name):
                continue

            # Convert genre_name -> label index
            if genre_name not in genre_to_idx:
                continue
            label_idx = genre_to_idx[genre_name]

            # Load the PNG
            img_path = os.path.join(folder_path, fname)
            try:
                img = Image.open(img_path).convert("L")
            except Exception as e:
                print(f"[ERROR] Failed to open image {img_path}: {e}")
                continue

            # Resize & normalize
            img = img.resize(IMG_SIZE, Image.BICUBIC)
            img_array = np.array(img, dtype=np.float32) / 255.0

            images.append(img_array)
            labels.append(label_idx)

            num_files_total += 1

            # Print a debug update every 500 files
            if num_files_total % 500 == 0:
                print(f"[DEBUG]   Processed {num_files_total} spectrograms so far...")

t1 = time.time()

print(f"[INFO] Loaded {num_files_total} spectrograms in total.")
print(f"[INFO] Elapsed time for loading: {t1 - t0:.2f} seconds")

# Convert to NumPy arrays
X = np.array(images, dtype=np.float32)  # shape: (N, 128, 128)
y = np.array(labels, dtype=np.int32)    # shape: (N,)

# -------------------------------------------------------------------
# 4) PREPARE DATA FOR TRAINING
# -------------------------------------------------------------------
num_classes = len(all_genres)
y_categorical = to_categorical(y, num_classes=num_classes)

# Reshape X to (N, 128, 128, 1)
X = np.expand_dims(X, axis=-1)

print("[INFO] Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)
print("Train set:", X_train.shape, y_train.shape)
print("Test  set: ", X_test.shape, y_test.shape)

# -------------------------------------------------------------------
# 5) DEFINE & BUILD A CNN MODEL
# -------------------------------------------------------------------
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("[INFO] Creating the CNN model...")
model = create_model()
model.summary()

# -------------------------------------------------------------------
# 6) TRAIN THE MODEL
# -------------------------------------------------------------------
print("[INFO] Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# -------------------------------------------------------------------
# 7) EVALUATE & PLOT
# -------------------------------------------------------------------
print("[INFO] Evaluating on test set...")
loss, acc = model.evaluate(X_test, y_test)
print(f"[RESULT] Test Accuracy: {acc*100:.2f}%")

plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training History')
plt.legend()
plt.show()

# -------------------------------------------------------------------
# 8) SAVE MODEL
# -------------------------------------------------------------------
model.save("genre_classification_model.h5")
print("[INFO] Model saved to genre_classification_model.h5")

# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Get the unique class indices in your test labels
unique_labels = np.unique(y_true)

# Build correct label names for just those present in the data
correct_target_names = [idx_to_genre[i] for i in unique_labels]

# Classification report with the correct label mapping
print("[INFO] Classification Report:")
report = classification_report(
    y_true,
    y_pred,
    labels=unique_labels,
    target_names=correct_target_names
)
print(report)