import os
import librosa.display
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import sys
import argparse

load_dotenv()
sample_rate = int(os.getenv('sample_rate'))
hop_length = int(os.getenv('hop_length'))
n_mels = int(os.getenv('n_mels'))
n_fft = int(os.getenv('n_fft'))
trim_duration = int(os.getenv('trim_duration'))
segment_length = int(os.getenv('segment_length'))


def normalize_audio(wav):
    # 標準化音頻信號
    if np.max(np.abs(wav)) == 0:
        return wav
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

def hamming_window(y):
    window = np.hamming(len(y))
    wav = y * window
    return wav

def trim_audio(wav, sr, trim_duration=0.5):
    # 移除前後 trim_duration 秒
    trim_samples = int(trim_duration * sr)
    if len(wav) > 2 * trim_samples:
        return wav[trim_samples:-trim_samples]
    else:
        return wav  # 如果音頻長度不足以移除前後 trim_duration 秒，則不進行裁剪

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
                    # 去除音頻兩端的靜音部分
                    y = librosa.effects.trim(y, top_db=amplification_factor, hop_length=hop_length)[0]
                    y = hamming_window(y)
                    path = root.split(os.path.sep)
                    filename = f'{path[-1]}_{file}'
                    pbar.set_postfix(file=filename, )
                    wav_files.append((y, filename))
                pbar.update(1)
    return wav_files


def convert_to_mel_spectrogram(y, n_fft, hop_length, n_mels, sr=16000):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
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


def segment_audio(wav_file, sr=44100, segment_length=2, verbose=False):
    """
    Segment audio into clips of a specified length.

    :param wav_file: Tuple containing the audio time series and file name
    :param sr: Sample rate
    :param segment_length: The desired length of each audio segment in seconds (default: 2)
    :param verbose: Print additional information (default: False)
    :return: A list of audio segments
    """
    # 載入音訊數據
    y, name = wav_file

    # 計算每個片段的樣本數
    segment_samples = int(segment_length * sr)

    # 切割音訊，每個片段長度固定為 segment_length 秒
    segments = []
    for start in range(0, len(y), segment_samples):
        end = start + segment_samples
        segment = y[start:end]

        # 如果片段長度不足 segment_length 秒，則跳過
        if len(segment) < segment_samples:
            continue

        segments.append(segment)

    if verbose:
        print(f"Segmented audio into {len(segments)} clips of {segment_length} seconds each.")

    return segments


def segment_files_and_save(files, sr, segment_length=2, output_dir='output', output_anomaly_dir='output_anomaly'):
    """
    Segment multiple audio files and save the segments.

    :param files: List of audio time series and file name tuples
    :param sr: Sample rate
    :param segment_length: The desired length of each audio segment in seconds (default: 2)
    :param output_dir: Directory to save the normal segments
    :param output_anomaly_dir: Directory to save the anomaly segments
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_anomaly_dir):
        os.makedirs(output_anomaly_dir)

    total_files = len(files)
    with tqdm(total=total_files, desc='Segment files', unit='file') as pbar:
        for i, wav in enumerate(files):
            segments = segment_audio(wav, sr=sr, segment_length=segment_length, verbose=False)
            main_file_name = os.path.splitext(os.path.basename(wav[1]))[0]
            pbar.set_postfix(file=f'{wav[1]}', segments=len(segments))
            for j, segment in enumerate(segments):
                file_name = f'{main_file_name}_segment_{j}.wav'
                if 'anomaly' in wav[1]:
                    sf.write(os.path.join(output_anomaly_dir, file_name), segment, sr)
                else:
                    sf.write(os.path.join(output_dir, file_name), segment, sr)
            pbar.update(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process wav files and optionally plot mel spectrograms.")
    parser.add_argument('directory', type=str, help="The path to the directory containing the wav files.")
    parser.add_argument('--plot', type=str, choices=['true', 'false'], default='false', help="Whether to plot mel spectrograms. Accepts 'true' or 'false'.")
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print("The provided directory does not exist.")
        sys.exit(1)

    wav_files = load_wav_files(sys.argv[1], target_sr=sample_rate, amplification_factor=80, trim_duration=trim_duration)
    plot = args.plot.lower() == 'true'

    if not os.path.exists('images/mel_spectrograms'):
        os.makedirs('images/mel_spectrograms')
    if (plot):
        for file in tqdm(wav_files, desc='Plotting mel spectrograms', unit='file'):
            wav, filename = file
            wav = convert_to_mel_spectrogram(wav, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sr=sample_rate)
            plot_mel_spectrogram(wav, hop_length=hop_length, sr=sample_rate, filename=filename, save_only=True)
    segment_files_and_save(wav_files, sr=sample_rate, segment_length=segment_length, output_dir='output',
                           output_anomaly_dir='output_anomaly')
