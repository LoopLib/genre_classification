# data_utils.py
import os
import pandas as pd

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

def load_metadata_and_filter(metadata_dir="data/fma_metadata", subset="large"):
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