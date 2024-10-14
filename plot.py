import os
import librosa.display
from dotenv import load_dotenv
from tqdm import tqdm
import soundfile as sf

import sys
import argparse

from conver_mel_spectrogram import compute_mel_spectrogram, plot_mel_spectrogram

load_dotenv()
sample_rate = int(os.getenv('sample_rate'))
n_mels = int(os.getenv('n_mels'))
segment_length = int(os.getenv('segment_length'))


# def highpass_filter(y, sr, cutoff=100, order=5):
#     """
#     Apply a high-pass filter to the audio signal to remove low-frequency noise like wind noise.
#
#     :param wav_data: Audio time series and file name tuple
#     :param sr: Sample rate
#     :param cutoff: Cutoff frequency for the high-pass filter
#     :param order: Order of the filter
#     :return: Filtered audio time series
#     """
#     nyquist = 0.5 * sr
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     y_filtered = filtfilt(b, a, y)
#     return y_filtered
#
# def lowpass_filter(y, sr, cutoff=100, order=5):
#     """
#     Apply a low-pass filter to the audio signal to remove high-frequency noise like hissing.
#
#     :param wav_data: Audio time series and file name tuple
#     :param sr: Sample rate
#     :param cutoff: Cutoff frequency for the low-pass filter
#     :param order: Order of the filter
#     :return: Filtered audio time series
#     """
#     nyquist = 0.5 * sr
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y_filtered = filtfilt(b, a, y)
#     return y_filtered

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


def convert_and_plot_mel_spectrogram(audio_data, sample_rate, filename=None, save_dir='images/mel_spectrograms',
                                     output_anomaly_dir='images/mel_spectrograms_anomaly'):
    mel_spectrogram_db, hop_length = compute_mel_spectrogram(audio_data, sample_rate=sample_rate, n_mels=n_mels, verbose=False)
    plot_mel_spectrogram(mel_spectrogram_db, hop_length=hop_length,
                         sample_rate=sample_rate,
                         filename=filename,
                         camp='coolwarm',
                         save_file=True,
                         save_dir=save_dir,
                         output_anomaly_dir=output_anomaly_dir)


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
    parser.add_argument('--plot', type=str, choices=['true', 'false'], default='false',
                        help="Whether to plot mel spectrograms. Accepts 'true' or 'false'.")
    parser.add_argument('image_dir', type=str, help="The path to the directory to save the mel spectrogram images.")
    parser.add_argument('--process', type=str, choices=['true', 'false'], default='true',
                        help="Whether to process the wav files. Accepts 'true' or 'false'.")
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

    if (plot):
        for file in tqdm(wav_files, desc='Plotting mel spectrograms', unit='file'):
            wav, filename = file
            convert_and_plot_mel_spectrogram(wav, sample_rate=sample_rate, filename=filename, save_dir=image_dir,
                                             output_anomaly_dir=image_dir + '_anomaly')
    if (process):
        segment_files_and_save(wav_files, sr=sample_rate, segment_length=segment_length, output_dir=sound_dir,
                               output_anomaly_dir=sound_dir + '_anomaly')
