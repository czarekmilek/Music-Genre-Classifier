import os
import librosa
import numpy as np
from sklearn.decomposition import PCA


def transform_file(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectrogram = np.abs(librosa.stft(y))

    mfcc_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    spectrogram_mean = np.mean(spectrogram, axis=1)

    combined_features = np.concatenate([mfcc_mean, chroma_mean, spectrogram_mean])

    return combined_features



def loadData(folder_path):
    features = []
    labels = []
    file_names = []

    for current_folder, _, files in os.walk(folder_path):
        for file in files:
            
            file_path = os.path.join(current_folder, file)
            parent_directory = os.path.dirname(file_path)

            features.append(transform_file(file_path))

            labels.append(parent_directory.split('/')[1].split('_')[0])

            file_names.append(file)

    return np.array(features), np.array(labels), file_names



# ======================================= PCA ==============================================

''' Shrinks number of features dimensions from around 1050 to lower numer,
    avoiding curse of dimensionality.
    Helpful especially when using KNN
'''

# PCA requires min(n_samples, n_features), but for KNN n_components=200 would be a better fit.
def reduce_dimensions(X, n_comp=0):
    n_components = n_comp if n_comp > 0 else min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced

# ==========================================================================================



def printFileData(X, y):
    print(f"\nKształt X: {X.shape} \n")
    print("=" * 100)
    for i, file_name in enumerate(file_names):
        print(f"\nCechy dla pliku {file_name}:")
        print(X[i], "\n")
        print("=" * 100)
    print(f"\nEtykiety: {y}\n")




if __name__ == "__main__":
    X, y, file_names = loadData(folder_path = "music_samples")
    X_reduced = reduce_dimensions(X)

    printFileData(X, y)
    print("="*50, "REDUCED", "="*50)
    printFileData(X_reduced, y)
