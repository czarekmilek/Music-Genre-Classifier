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


X = np.array([])
y = np.array([]) 
folder_path = "music_samples"
file_names = []

for current_folder, sub_folder, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(current_folder, file) 
        parent_directory = os.path.dirname(file_path)

        new_arr = transform_file(file_path)

        if X.size == 0:
            X = new_arr.reshape(1, -1)
        else:
            X = np.vstack([X, new_arr]) 

        y = np.append(y, parent_directory.split('/')[1].split('_')[0]) # adding labels to y
        file_names.append(file)





# ======================================= PCA ==============================================

''' Shrinks number of features dimensions from around 1050 to lower numer,
    avoiding curse of dimensionality.
    Helpful especially when using KNN
'''

# PCA requires min(n_samples, n_features), but for KNN n_components=200 would be a better fit.
pca = PCA(n_components=min(X.shape[0], X.shape[1])) 
X_reduced = pca.fit_transform(X)

# ==========================================================================================



def printFileData(X, y):
    print(f"\nKształt X: {X.shape} \n")
    print("=" * 100)
    for i, file_name in enumerate(file_names):
        print(f"\nCechy dla pliku {file_name}:")
        print(X[i], "\n")
        print("=" * 100)
    print(f"\nEtykiety: {y}\n")


printFileData(X, y)
print("="*50, "REDUCED", "="*50)
printFileData(X_reduced, y)