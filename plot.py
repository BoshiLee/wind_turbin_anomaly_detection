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
from scipy.fft import rfft
from scipy.signal import windows

load_dotenv()
sample_rate = int(os.getenv('sample_rate'))
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

def lowpass_filter(y, sr, cutoff=100, order=5):
    """
    Apply a low-pass filter to the audio signal to remove high-frequency noise like hissing.

    :param wav_data: Audio time series and file name tuple
    :param sr: Sample rate
    :param cutoff: Cutoff frequency for the low-pass filter
    :param order: Order of the filter
    :return: Filtered audio time series
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y_filtered = filtfilt(b, a, y)
    return y_filtered

def load_wav_files(directory, target_sr=16000):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        with tqdm(total=len(files), desc='Loading files', unit='file') as pbar:
            for file in files:
                if file.endswith(".wav") and file != 'all_channel.wav':
                    file_path = os.path.join(root, file)
                    y, sr = librosa.load(file_path, sr=None)
                    if sr != target_sr:
                        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                    path = root.split(os.path.sep)
                    filename = f'{path[-1]}_{file}'
                    pbar.set_postfix(file=filename, )
                    wav_files.append((y, filename))
                pbar.update(1)
    return wav_files

def stft(x, n_fft, hop_length, window):
    num_frames = 1 + (len(x) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(x, shape=(n_fft, num_frames),
                                             strides=(x.itemsize, hop_length*x.itemsize))
    return rfft(frames * window[:, None], n=n_fft, axis=0)

def convert_to_spectrogram(audio_data, sample_rate, filename=None, save_dir='images/stft_spectrograms'):
    # 確保音頻數據是單聲道的
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]

    # 設置STFT參數
    n_fft = sample_rate // 5
    hop_length = n_fft // 4  # 75% 重疊

    # 應用漢寧窗
    window = windows.hann(n_fft, sym=False)

    # 執行STFT
    stft_result = stft(audio_data, n_fft, hop_length, window)

    # 計算頻譜圖（幅度譜）
    spectrogram = np.abs(stft_result)

    # 轉換為分貝刻度
    spectrogram_db = 20 * np.log10(spectrogram + 1e-10)

    # 創建時間軸和頻率軸
    time = np.arange(spectrogram.shape[1]) * hop_length / sample_rate
    freq = np.linspace(0, sample_rate / 2, spectrogram.shape[0])

    # 繪製頻譜圖
    plt.figure(figsize=(15, 10))
    plt.imshow(spectrogram_db, aspect='auto', origin='lower',
               extent=[time.min(), time.max(), freq.min(), freq.max()],
               cmap='jet')

    plt.colorbar(label='amplitude (dB)')
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.title(f'Spectrogram of Audio Signal (n_fft = {n_fft})')

    # 設置y軸為對數刻度
    plt.yscale('log')
    plt.ylim(20, sample_rate / 2)  # 限制y軸範圍從20Hz到奈奎斯特頻率

    # 設置y軸刻度
    plt.yticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
               ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename}_spectrogram.png')
    plt.close()


def plot_mel_spectrogram(audio_data, sample_rate, filename=None, save_dir='images/mel_spectrograms'):
    # 確保音頻數據是單聲道的
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]

    # 將音頻數據轉換為浮點型並歸一化
    if audio_data.dtype.kind in 'iu':
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    elif audio_data.dtype.kind == 'f':
        audio_data = audio_data.astype(np.float32)
        max_value = np.max(np.abs(audio_data))
        if max_value > 1.0:
            audio_data /= max_value

    # 設置STFT參數
    n_fft = sample_rate // 5
    hop_length = n_fft // 4  # 75% 重疊

    # 應用漢寧窗
    window = windows.hann(n_fft, sym=False)

    # 執行STFT
    stft_result = stft(audio_data, n_fft, hop_length, window)

    # 計算功率譜
    power_spectrum = np.abs(stft_result) ** 2

    # 創建梅爾濾波器組
    n_mels = 128
    mel_filterbank = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)

    # 將功率譜轉換為梅爾頻譜
    mel_spectrogram = np.dot(mel_filterbank, power_spectrum)

    # 轉換為分貝刻度
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 創建時間軸
    time = np.arange(mel_spectrogram.shape[1]) * hop_length / sample_rate

    # 繪製梅爾頻譜圖
    plt.figure(figsize=(15, 10))
    plt.imshow(mel_spectrogram_db, aspect='auto', origin='lower',
               extent=[time.min(), time.max(), 0, n_mels],
               cmap='jet')

    plt.colorbar(label='amplitude (dB)')
    plt.xlabel('time (sec)')
    plt.ylabel('mel frequency')
    plt.title(f'Mel Spectrogram of Audio Signal (n_fft = {n_fft}, n_mels = {n_mels})')
    plt.savefig(f'{save_dir}/{filename}_mel_spectrogram.png')

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
    parser.add_argument('image_dir', type=str, help="The path to the directory to save the mel spectrogram images.")
    parser.add_argument('--process', type=str, choices=['true', 'false'], default='true', help="Whether to process the wav files. Accepts 'true' or 'false'.")
    parser.add_argument('output_dir', type=str, help="The path to the directory to save the processed wav")
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print("The provided directory does not exist.")
        sys.exit(1)

    wav_files = load_wav_files(sys.argv[1], target_sr=sample_rate)
    plot = args.plot.lower() == 'true'
    process = args.process.lower() == 'true'
    sound_dir = f'audio/{args.output_dir}'
    image_dir = f'images/{args.image_dir}'

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if (plot):
        for file in tqdm(wav_files, desc='Plotting mel spectrograms', unit='file'):
            wav, filename = file
            plot_mel_spectrogram(wav, sample_rate=sample_rate, filename=filename, save_dir=image_dir)
    if (process):
        segment_files_and_save(wav_files, sr=sample_rate, segment_length=segment_length, output_dir=sound_dir,
                               output_anomaly_dir=sound_dir + '_anomaly')
