# Music Genre Classification Pipeline

This repository provides tools and scripts to perform music genre classification on the Free Music Archive (FMA) dataset. It includes utilities for data loading and verification, spectrogram generation, feature extraction, and model training (Random Forest and Convolutional Neural Networks).

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Spectrogram Extraction](#spectrogram-extraction)
- [Feature Extraction](#feature-extraction)
- [Training and Evaluation](#training-and-evaluation)
  - [Random Forest Classifier](#random-forest-classifier)
  - [CNN on Audio Features](#cnn-on-audio-features)
  - [CNN on Spectrogram Images](#cnn-on-spectrogram-images)
- [Scripts and Modules](#scripts-and-modules)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```
├── data_utils.py           # Metadata loading & file-check utilities
├── extract_spectrogram.py  # Convert audio to mel-spectrogram images
├── feature_extraction.py   # Extract MFCCs and spectral features from audio
├── genre_classification.py # End-to-end training script (RF & CNN-features)
├── cnn_features.py         # Build CNN model for feature input
├── cnn_spectrogram.py      # Train CNN directly on spectrogram images
├── rf.py                   # Random Forest training & metric plotting
├── data/                   # Raw audio and metadata
│   ├── fma_metadata/       # FMA metadata CSVs
│   ├── fma_small/          # Subset of audio files (.mp3)
│   └── fma_small_spectrograms/ # Generated spectrogram PNGs
└── requirements.txt        # Python dependencies
```

## Prerequisites
- Python 3.7+  
- FMA small or large dataset downloaded locally  
- Recommended RAM: ≥8GB  

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux / macOS
   venv\\Scripts\\activate     # Windows
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation
1. Download FMA dataset (small or large) following instructions at https://github.com/mdeff/fma.
2. Place the audio files under `data/fma_small/` (or `data/fma_large/`) preserving the directory structure.
3. Ensure metadata CSVs are in `data/fma_metadata/`.

## Spectrogram Extraction
Use `extract_spectrogram.py` to convert each audio file into a mel-spectrogram PNG:
```bash
python extract_spectrogram.py
```
- Output images are saved to `data/fma_small_spectrograms/<genre>/<track_id>.png`.
- Parameters such as `dpi`, `n_mels`, and `hop_length` can be adjusted in the script.

## Feature Extraction
The `feature_extraction.py` module provides `feature_extraction(df, n_mfcc, track_dir)` to extract:
- MFCC mean, std, delta, delta-delta
- RMS energy
- Spectral centroid, bandwidth, rolloff, contrast
- Zero-crossing rate
- Chroma and Tonnetz features
- Spectral flatness

Example usage (internal):
```python
from feature_extraction import feature_extraction
X, valid_idx = feature_extraction(df_tracks, n_mfcc=40, track_dir="data/fma_small")
```

## Training and Evaluation
### Random Forest Classifier
- Uses `models/rf.py` → `train_rf_model(...)`.  
- Trains `RandomForestClassifier` with hyperparameters:  
  - `n_estimators=500`, `max_depth=30`, etc.  
- Outputs:  
  - Saved model (`rf.joblib`),  
  - Label encoder & scaler,  
  - `predictions.txt` sample,  
  - `rf_model_metrics.png` bar chart.  

### CNN on Audio Features
- Builds simple CNN in `cnn_features.py` via `build_cnn_model(input_shape, num_classes)`.  
- Can be integrated in `genre_classification.py` by reshaping feature arrays and calling `train_cnn_model(...)`.

### CNN on Spectrogram Images
- `cnn_spectrogram.py` scans `data/fma_small_spectrograms/`, loads and preprocesses PNGs, splits into train/test, builds & trains CNN:
  - Conv2D → MaxPooling → Flatten → Dense → Dropout → Softmax
  - Evaluates accuracy, plots training history, prints classification report.

## Scripts and Modules
- **data_utils.py**  
  - `load_metadata_and_filter(metadata_dir, subset)`: returns DataFrame with `track_id`, `genre_top`, `path`.  
  - `check_missing_files(df, track_dir)`: reports missing `.mp3` files.  

- **extract_spectrogram.py**  
  - `extract_spectrogram(audio_path, save_dir, genre, dpi, n_mels, hop_length)`  
  - `process_all_audio_files(audio_dir, spectrogram_dir, ...)`  

- **feature_extraction.py**  
  - `feature_extraction(df, n_mfcc, track_dir)` → `(X_features, valid_indices)`  

- **genre_classification.py**  
  - End-to-end pipeline: loads metadata, merges precomputed features, splits data, scales, trains RF & CNN-features models.  

- **cnn_features.py**  
  - `build_cnn_model(input_shape, num_classes)` → compiled Keras model.  

- **cnn_spectrogram.py**  
  - Script to train CNN directly on spectrogram images.  

- **rf.py**  
  - `train_rf_model(...)`: trains RF, saves artifacts, plots metrics.  

## Usage Examples
```bash
# 1. Generate spectrograms
python extract_spectrogram.py

# 2. Train RF & CNN-features models
python genre_classification.py

# 3. Train CNN on spectrograms
python cnn_spectrogram.py
```  



