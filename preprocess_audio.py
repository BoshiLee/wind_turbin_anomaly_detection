import os
import librosa.display
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import sys
def normalize_audio(wav):
    # 標準化音頻信號
    return wav / np.max(np.abs(wav))

def loudness_normalize_audio(wav, target_dB=-20.0):
    # 計算信號的RMS
    rms = np.sqrt(np.mean(np.square(wav)))
    # 防止RMS為零的情況
    if rms == 0:
        rms = 1e-10  # 使用一個非常小的值來避免除以零
    # 計算響度校正因子
    scalar = 10 ** (target_dB / 20) / rms
    return wav * scalar


def amplify_audio(wav, factor):
    # 放大音頻信號
    return wav * factor

def trim_audio(wav, sr, trim_duration=0.5):
    # 移除前後 trim_duration 秒
    trim_samples = int(trim_duration * sr)
    if len(wav) > 2 * trim_samples:
        return wav[trim_samples:-trim_samples]
    else:
        return wav  # 如果音頻長度不足以移除前後 trim_duration 秒，則不進行裁剪


def load_wav_files(directory, target_sr=16000, amplification_factor=80, trim_duration=1):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        with tqdm(total=len(files), desc='Loading files', unit='file') as pbar:
            for file in files:
                if file.endswith(".wav") and file != 'all_channel.wav':
                    file_path = os.path.join(root, file)
                    y, sr = librosa.load(file_path, sr=None)
                    if sr != target_sr:
                        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

                    y = trim_audio(y, sr=target_sr, trim_duration=trim_duration)
                    y = loudness_normalize_audio(y)
                    y = amplify_audio(y, amplification_factor)
                    y = highpass_filter(y, sr=target_sr, cutoff=4096, order=5)
                    y = convert_to_mel_spectrogram(y, n_fft=512, hop_length=256, n_mels=64, sr=target_sr)

                    # remove wav_directory from root
                    path = root.split(os.path.sep)
                    filename = f'{path[1]}_{file}'
                    pbar.set_postfix(file=filename, )
                    wav_files.append((y, filename))
                pbar.update(1)
    return wav_files


def highpass_filter(y, sr, cutoff=100, order=5):
    """
    Apply a high-pass filter to the audio signal to remove low-frequency noise like wind noise.

    :param wav_data: Audio time series and file name tuple
    :param sr: Sample rate
    :param cutoff: Cutoff frequency for the high-pass filter
    :param order: Order of the filter
    :return: Filtered audio time series
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = filtfilt(b, a, y)
    return y_filtered

def convert_to_mel_spectrogram(y, n_fft, hop_length, n_mels, sr=16000):
    window = np.hamming(len(y))
    audio = y * window
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                     n_mels=n_mels, fmin=25)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def plot_mel_spectrogram(mel_spectrogram, hop_length=512, sr=16000, filename='', save_only=False):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel spectrogram: {filename}')
    plt.savefig(f'images/mel_spectrograms/{filename}_mel_spectrogram.png')
    if not save_only:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 1 :
        print("Please provide the path to the directory containing the wav files.")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print("The provided directory does not exist.")
        sys.exit(1)
    sample_rate = sys.argv[2] if len(sys.argv) > 2 else 44100
    wav_files = load_wav_files(sys.argv[1], target_sr=sample_rate, amplification_factor=80, trim_duration=1)

    for wav, filename in tqdm(wav_files, desc='Plotting mel spectrograms', unit='file'):
        plot_mel_spectrogram(wav, hop_length=512, sr=sample_rate, filename=filename, save_only=True)
