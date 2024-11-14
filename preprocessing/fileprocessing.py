import os
import numpy as np
import librosa
import scipy


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 위의 주석은 스크립트 버전에서, 아래 버전은 colab에서 사용
BASE_DIR = os.getcwd()
TRAIN_DIR = os.path.join(BASE_DIR, '/content/drive/MyDrive/signal&system/train')
TEST_DIR = os.path.join(BASE_DIR, '/content/drive/MyDrive/signal&system/test')
RAWTRAIN_DIR = os.path.join(BASE_DIR, '/content/drive/MyDrive/signal&system/rawtrain')
RAWTEST_DIR = os.path.join(BASE_DIR, '/content/drive/MyDrive/signal&system/rawtest')

# 모든 하위 디렉터리까지 파일 탐색
def find_audio_files(input_dir):
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.ogg')):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)
                print(f"Audio file found: {file_path}")
    return audio_files

# Fourier Transform 함수
def perform_fourier_transform_n_save_npy_file(file_path, output_path, FS=16000):
    try:
        print(f"Loading audio file: {file_path}")
        data, sample_rate = librosa.load(file_path, sr=None)
        if data is None or len(data) == 0:
            print(f"Failed to load audio (empty data): {file_path}")
            return None
        print(f"Audio loaded successfully: {file_path}, Sample rate: {sample_rate}, Data length: {len(data)}")

        if sample_rate != 16000:
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=FS)


        _, _, freq_data = scipy.signal.stft(data, fs=16000, window='rectangular', nperseg=512, noverlap=0, nfft=512)
        magnitude = np.abs(freq_data)  # [fft bin, frame length]

        np.save(output_path, magnitude)
    except Exception as e:
        print(f"Error during file processing: {file_path}, Error message: {str(e)}")
        return None

# 오디오 파일 처리 함수
def process_audio_files(input_dir, output_dir):
    audio_files = find_audio_files(input_dir)
    for file_path in audio_files:
        relative_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, relative_path).replace(relative_path[-4:], '.npy')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        perform_fourier_transform_n_save_npy_file(file_path, output_path)

# 메인 함수
def main():
    os.makedirs(RAWTRAIN_DIR, exist_ok=True)
    os.makedirs(RAWTEST_DIR, exist_ok=True)

    print("Processing train data...")
    process_audio_files(TRAIN_DIR, RAWTRAIN_DIR)

    print("Processing test data...")
    process_audio_files(TEST_DIR, RAWTEST_DIR)

    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()



