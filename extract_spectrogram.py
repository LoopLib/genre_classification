import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_spectrogram(audio_path, save_dir, genre, dpi=300, n_mels=512, hop_length=512):
    """
    Converts an audio file into a high-quality mel spectrogram and saves it within a genre-specific folder.
    
    Parameters:
      audio_path (str): Path to the input audio file.
      save_dir (str): Base directory to save spectrogram images.
      genre (str): Genre label (used for folder categorization).
      dpi (int): Dots per inch (resolution) for the saved image. Higher values give better quality.
      n_mels (int): Number of Mel bands for the spectrogram.
      hop_length (int): Hop length for time resolution.
    """
    try:
        # Load audio file (using a consistent sample rate and mono conversion)
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        if len(y) == 0:
            print(f"⚠ Warning: Empty audio file {audio_path}")
            return

        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Ensure the genre-specific directory exists
        genre_dir = os.path.join(save_dir, genre)
        os.makedirs(genre_dir, exist_ok=True)

        # Create a high-quality spectrogram figure
        # Adjust figsize as needed; here 12x4 inches gives a detailed image at 300 DPI (approx 3600x1200 pixels)
        plt.figure(figsize=(12, 4), dpi=dpi)
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, 
                                 x_axis='time', y_axis='mel', fmax=8000, cmap='inferno')
        plt.axis('off')  # Remove axes for a clean image

        # Define the filename and save path
        filename = os.path.splitext(os.path.basename(audio_path))[0] + ".png"
        save_path = os.path.join(genre_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"✅ Spectrogram saved: {save_path}")
    except Exception as e:
        print(f"⚠ Error processing {audio_path}: {e}")

def process_all_audio_files(audio_dir, spectrogram_dir, dpi=300, n_mels=512, hop_length=512):
    """
    Processes all audio files in the specified directory, organized by genre,
    and exports high-quality spectrogram images.
    
    Parameters:
      audio_dir (str): Path containing genre subdirectories with audio files.
      spectrogram_dir (str): Base directory where spectrogram images will be saved.
      dpi (int): Resolution for the exported images.
      n_mels (int): Number of Mel bands.
      hop_length (int): Hop length for the spectrogram.
    """
    os.makedirs(spectrogram_dir, exist_ok=True)
    
    for genre in os.listdir(audio_dir):
        genre_path = os.path.join(audio_dir, genre)
        if os.path.isdir(genre_path):
            print(f"Processing genre: {genre}")
            for file in os.listdir(genre_path):
                if file.lower().endswith(".mp3"):
                    audio_path = os.path.join(genre_path, file)
                    extract_spectrogram(audio_path, spectrogram_dir, genre, dpi=dpi, n_mels=n_mels, hop_length=hop_length)

if __name__ == "__main__":
    AUDIO_DIR = "data/fma_small"
    SPECTROGRAM_DIR = "data/fma_small_spectrograms"
    process_all_audio_files(AUDIO_DIR, SPECTROGRAM_DIR)
