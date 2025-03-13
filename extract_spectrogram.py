import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define spectrogram extraction function with improvements
def extract_spectrogram(audio_path, save_dir, img_size=(512, 512), n_mels=512, hop_length=512):
    """
    Converts an audio file into a mel spectrogram and saves it with high quality.
    - audio_path: Path to input audio file
    - save_dir: Where to save the spectrogram
    - img_size: Resize dimensions for final spectrogram image
    - n_mels: Number of Mel bands for the spectrogram
    - hop_length: Hop length for better time resolution
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        if len(y) == 0:
            print(f"⚠ Warning: Empty audio file {audio_path}")
            return

        # Generate Mel Spectrogram with higher resolution
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000, hop_length=hop_length)
        
        # Convert the Mel spectrogram to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[0] == 0 or mel_spec_db.shape[1] == 0:
            print(f"⚠ Warning: Spectrogram generation failed for {audio_path}")
            return

        # Resize spectrogram for consistency
        mel_spec_db_resized = cv2.resize(mel_spec_db, img_size)

        # Normalize the spectrogram for better contrast
        mel_spec_db_resized = np.clip(mel_spec_db_resized, a_min=0, a_max=255)  # Clip values
        mel_spec_db_resized = mel_spec_db_resized / 255.0  # Normalize to [0, 1]

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the spectrogram as an image using a more perceptually accurate color map
        filename = os.path.splitext(os.path.basename(audio_path))[0] + ".png"
        save_path = os.path.join(save_dir, filename)
        plt.imsave(save_path, mel_spec_db_resized, cmap='inferno')  # Change to 'inferno' for better color mapping

        if os.path.exists(save_path):
            print(f"✅ Spectrogram saved: {save_path}")
        else:
            print(f"⚠ Error: Spectrogram was NOT saved for {audio_path}")

    except Exception as e:
        print(f"⚠ Error processing {audio_path}: {e}")

# Main function to extract spectrograms for all audio files
def process_all_audio_files(audio_dir, spectrogram_dir):
    """
    Extracts high-quality spectrograms from all audio files in the dataset.
    - audio_dir: Path to directory containing audio files
    - spectrogram_dir: Where to save spectrograms
    """
    os.makedirs(spectrogram_dir, exist_ok=True)

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".mp3"):
                audio_path = os.path.join(root, file)
                extract_spectrogram(audio_path, spectrogram_dir)

# Run script
if __name__ == "__main__":
    AUDIO_DIR = "data/fma_small"
    SPECTROGRAM_DIR = "data/fma_small_spectrograms"
    process_all_audio_files(AUDIO_DIR, SPECTROGRAM_DIR)
