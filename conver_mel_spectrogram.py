import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import windows, stft
from sympy import true


def compute_mel_spectrogram(audio_data, sample_rate, n_mels=128, n_fft=2048, hop_length=512, verbose=False):
    # 確保音頻數據是單聲道的
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]

    # 將音頻數據轉換為浮點型並正規化
    if audio_data.dtype.kind in 'iu':
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    elif audio_data.dtype.kind == 'f':
        audio_data = audio_data.astype(np.float32)
        max_value = np.max(np.abs(audio_data))
        if max_value > 1.0:
            audio_data /= max_value

    # 設置STFT參數
    n_fft = n_fft if n_fft else int(sample_rate // 5)
    hop_length = hop_length if hop_length else n_fft // 4

    # 應用漢寧窗
    window = windows.hann(n_fft, sym=False)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     window='hann',
                                                     n_mels=n_mels)

    # 轉換為分貝刻度
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if verbose:
        print(f"STFT窗口大小: {n_fft}")
        print(f"音頻採樣率: {sample_rate} Hz")
        print(f"STFT重疊: {hop_length}")
        print(f"時間分辨率: {hop_length / sample_rate:.3f} 秒")
        print(f"頻率分辨率: {sample_rate / n_fft:.2f} Hz")
        print(f"音頻長度: {len(audio_data) / sample_rate:.2f} 秒")

    return mel_spectrogram_db, hop_length


def plot_mel_spectrogram(mel_spectrogram_db, hop_length, sample_rate, filename=None, save_file=True,
                         save_dir='images/mel_spectrograms', output_anomaly_dir='images/mel_spectrograms_anomaly', camp='jet'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(output_anomaly_dir):
        os.makedirs(output_anomaly_dir)

    # 創建時間軸
    time = np.arange(mel_spectrogram_db.shape[1]) * hop_length / sample_rate

    # 繪製梅爾頻譜圖
    plt.figure(figsize=(15, 10))
    plt.imshow(mel_spectrogram_db, aspect='auto', origin='lower',
               extent=[time.min(), time.max(), 0, mel_spectrogram_db.shape[0]],
               cmap=camp)

    plt.colorbar(label='amplitude (dB)')
    plt.xlabel('time (sec)')
    plt.ylabel('mel frequency')

    plt.xticks(np.arange(0, time.max(), 1))

    # 設置y軸刻度
    mel_ticks = librosa.mel_frequencies(n_mels=mel_spectrogram_db.shape[0], fmin=0, fmax=sample_rate / 2)
    plt.yticks(np.arange(0, mel_spectrogram_db.shape[0], mel_spectrogram_db.shape[0] // 10),
               [f'{int(f)}' for f in mel_ticks[::mel_spectrogram_db.shape[0] // 10]])

    plt.tight_layout()
    plt.title(f'Mel Spectrogram of Audio Signal (n_mels = {mel_spectrogram_db.shape[0]})')
    if save_file:
        if 'anomaly' in filename:
            plt.savefig(f'{output_anomaly_dir}/{filename}_mel_spectrogram.png')
        else:
            plt.savefig(f'{save_dir}/{filename}_mel_spectrogram.png')
        plt.close()
    else:
        plt.show()


def compute_stft_spectrogram(audio_data, sample_rate,  n_fft=2048, hop_length=512, verbose=False):
    # 確保音頻數據是單聲道的
    # 確保音頻數據是單聲道的
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]

    # 設置STFT參數
    n_fft = n_fft if n_fft else int(sample_rate // 5)
    hop_length = hop_length if hop_length else n_fft // 4

    # 應用漢寧窗
    window = windows.hann(n_fft, sym=False)

    # 執行STFT
    f, t, stft_result = stft(audio_data, nperseg=n_fft, noverlap=hop_length, window=window)

    # 計算頻譜圖（幅度譜）
    spectrogram = np.abs(stft_result)

    # 轉換為分貝刻度
    spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
    if verbose:
        print(f"STFT窗口大小: {n_fft}")
        print(f"音頻採樣率: {sample_rate} Hz")
        print(f"STFT重疊: {hop_length}")
        print(f"時間分辨率: {hop_length / sample_rate:.3f} 秒")
        print(f"頻率分辨率: {sample_rate / n_fft:.2f} Hz")
        print(f"音頻長度: {len(audio_data) / sample_rate:.2f} 秒")

    return spectrogram, hop_length


def plot_stft_spectrogram(spectrogram, hop_length, sample_rate, n_fft=2048, filename=None, save_file=True,
                          save_dir='images/stft_spectrograms', output_anomaly_dir='images/stft_spectrograms_anomaly', camp='jet'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(output_anomaly_dir):
        os.makedirs(output_anomaly_dir)

    # 轉換為分貝刻度
    spectrogram_db = 20 * np.log10(spectrogram + 1e-10)

    # 創建時間軸和頻率軸
    time = np.arange(spectrogram.shape[1]) * hop_length / sample_rate
    freq = np.linspace(0, sample_rate / 2, spectrogram.shape[0])


    # 繪製頻譜圖
    plt.figure(figsize=(15, 10))
    plt.imshow(spectrogram_db, aspect='auto', origin='lower',
               extent=[time.min(), time.max(), freq.min(), freq.max()],
               cmap=camp)

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

    if save_file:
        if 'anomaly' in filename:
            plt.savefig(f'{output_anomaly_dir}/{filename}_stft_spectrogram.png')
        else:
            plt.savefig(f'{save_dir}/{filename}_stft_spectrogram.png')
        plt.close()

    else:
        plt.show()
