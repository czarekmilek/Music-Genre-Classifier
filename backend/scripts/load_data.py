import numpy as np
import os
from scripts.extract_features import extract_audio_features
import pandas as pd
from tqdm import tqdm


FOLDER_PATH = "../data/raw/music"

def loadData():
    all_features = []
    
    for current_folder, _, files in tqdm(os.walk(FOLDER_PATH)):
        for file in files:
            
            file_path = os.path.join(current_folder, file)
            parent_directory = os.path.dirname(file_path)
            if not file.startswith("._"):
                try: 
                    features = extract_audio_features(file_path=file_path)

                    features["title"] = file
                    features["category"] = parent_directory.split("/")[-1]
                
                    all_features.append(features)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

    return pd.DataFrame(all_features)

if __name__ == "__main__":
    dataset = loadData()
    
    # Save to CSV
    dataset.to_csv("../data/processed/music_features.csv", index=False)
    
    print("Features extracted:")
    print("\nShape:", dataset.shape)
    print("\nColumns:", dataset.columns.tolist())
