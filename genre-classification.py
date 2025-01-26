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

