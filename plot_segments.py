import os
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import sys
# %%
target_sample_rate = 44100
hop_length = 512
n_mels = 256
n_fft = 2048

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))


def convert_to_mel_spectrogram(audio, n_fft, hop_length, n_mels, sr=44100):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                     n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db


def plot_mel_spectrogram(mel_spectrogram, sr=44100, hop_length=512, filename: str = None):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel spectrogram of {filename}')
    plt.savefig(f'mel_spectrograms_seg_anomaly/{filename}_mel_spectrogram.png')
    plt.close()


# pass directory where segmented audio files are stored from command line
def load_segmented_files(args):
    if len(args) < 1:
        print("Usage: python plot_segments.py <directory>")
        sys.exit(1)
    wav_files = []
    directory = args[0]
    for file in tqdm(os.listdir(directory)):
        if file.endswith(".wav"):
            file_path = os.path.join(directory, file)
            y, sr = librosa.load(file_path, sr=None)
            name = file_path.split(os.path.sep)
            name = name[-1]

            y = normalize_audio(y)
            y = convert_to_mel_spectrogram(y, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sr=sr)

            plot_mel_spectrogram(y, filename=name, sr=sr, hop_length=hop_length)
            wav_files.append(y)
    return wav_files


load_segmented_files(sys.argv[1:])
