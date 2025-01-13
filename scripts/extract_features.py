import librosa
import pandas as pd
import numpy as np
from pathlib import Path


"""
    Extracts data from provided mp3 file using librosa library
    sr - sampling rate z pliku
    y - wektor z zsamplowanymi danymi
    
    Feature Explanations:
    1. Temporal Features:
       - duration: Length of the track in seconds
       - zero_crossing_rate: Rate at which signal changes from positive to negative
         High for noisy/percussive sounds, low for harmonic sounds
    
    2. Spectral Features:
       - spectral_centroid: Center of mass of the spectrum
         Higher values → brighter/sharper sound (like cymbals)
         Lower values → deeper/bass-heavy sound
       - spectral_rolloff: Frequency below which 85% of signal energy lies
         Helps distinguish voiced/unvoiced sounds
       - spectral_bandwidth: Width of frequency band containing most energy
         Wide → noisy/complex sound
         Narrow → pure/simple tones
    
    3. Rhythm Features:
       - tempo: Beats per minute (BPM)
         Distinguishes between slow and fast songs
    
    4. MFCC (Mel-frequency cepstral coefficients):
       - 13 coefficients capturing the overall shape of spectral envelope
       - mfcc_1: Overall energy/loudness
       - mfcc_2-13: Timbre and phonetic content
       - Both mean and std are captured to represent average and variation
    
    5. Chroma Features:
       - Represents the 12 pitch classes in music
       - High values indicate dominant notes/keys
       - Useful for detecting harmony and key changes
    
    6. RMS Energy:
       - Root Mean Square energy
       - Represents perceived loudness/volume
       - Higher values → louder sections
       - Variation (std) indicates dynamic range
"""
def extract_audio_features(file_path: str, sr=22050):

    y, sr = librosa.load(file_path, sr=sr)

    features = {}
    
    # 1. Temporal Features
    features['duration'] = librosa.get_duration(y=y, sr=sr)
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # 2. Spectral Features
    # Spectral Centroid - brightness of sound
    features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_centroid_std'] = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Spectral Rolloff - shape of signal
    features['spectral_rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Spectral Bandwidth
    features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    
    # 3. Rhythm Features
    # Tempo and Beat Features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    
    # 4. MFCC (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfcc)
        features[f'mfcc_{i+1}_std'] = np.std(mfcc)
    
    # 5. Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    
    # 6. RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    return features