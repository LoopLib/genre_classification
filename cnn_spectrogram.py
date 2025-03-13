import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from data_utils import load_metadata_and_filter, check_missing_files

# Define constants
IMG_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 32
EPOCHS = 30
SPECTROGRAM_DIR = "spectrograms/"  # Directory where spectrograms are stored

def load_spectrograms(df_tracks):
    """Loads spectrogram images based on track IDs and returns image arrays and corresponding labels."""
    images = []
    labels = []
    
    for _, row in df_tracks.iterrows():
        track_id = row['track_id']
        genre = row['genre_top']
        img_path = os.path.join(SPECTROGRAM_DIR, f"{track_id}.png")
        
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            labels.append(genre)
        
    return np.array(images), np.array(labels)

def build_efficientnet_model(num_classes):
    """Builds a transfer learning model using EfficientNetB0."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False  # Freeze the base model initially
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def fine_tune_model(model):
    """Fine-tunes the last few layers of the base model."""
    base_model = model.layers[0]
    base_model.trainable = True  # Unfreeze the base model
    
    # Fine-tune only the last few layers of the base model
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=1e-5),  # Use a lower learning rate for fine-tuning
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    subset = "small"  # Change to 'large' if needed
    print(f"Loading metadata for subset='{subset}'...")
    df_tracks = load_metadata_and_filter(metadata_dir="data/fma_metadata", subset=subset)
    df_tracks.dropna(inplace=True)
    check_missing_files(df_tracks, track_dir="data/fma_large")
    
    # Load spectrogram images and labels
    X, y = load_spectrograms(df_tracks)
    
    if len(X) == 0 or len(y) == 0:
        print("No valid spectrograms found. Exiting...")
        return
    
    # Encode labels
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_categorical, test_size=0.3, random_state=42, stratify=y_categorical)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
    
    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    # Learning Rate Scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    # Build and train model
    model = build_efficientnet_model(num_classes=y_categorical.shape[1])
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=10,  # Train for 10 epochs initially
        callbacks=[reduce_lr]
    )
    
    # Fine-tune the model
    model = fine_tune_model(model)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=20,  # Fine-tune for 20 more epochs
        callbacks=[reduce_lr]
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model.save("efficientnet_genre_classification.h5")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()