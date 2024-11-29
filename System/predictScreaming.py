import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from torch import nn
import sounddevice as sd
from scipy.signal import stft
from google.cloud import speech

# 탐지할 키워드 정의
keywords = ["help", "please help", "help me", "도와주어", "도와주세요"]

# JSON 키 파일 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/iuseong/Downloads/signal & system/methodical-tea-442914-r5-3dca061d585f.json"

# Google Speech-to-Text 클라이언트 초기화
client = speech.SpeechClient()

# 음성 녹음 함수
def record_audio(duration=2, sampling_rate=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()
    file_path = "recorded_audio.wav"
    sf.write(file_path, audio_data, sampling_rate)
    print("Recording saved.")
    return file_path, sampling_rate

# Google Speech-to-Text API 함수
def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",  # 기본 언어
            alternative_language_codes=["ko-KR"]  # 보조 언어 설정
        )

        response = client.recognize(config=config, audio=audio)
        transcribed_text = " ".join([result.alternatives[0].transcript for result in response.results])
        return transcribed_text

    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""  # 빈 문자열 반환

# 키워드 탐지 함수
def detect_keywords(text):
    print(f"Transcribed text: {text}")
    for keyword in keywords:
        if keyword.lower() in text.lower():
            print(f"Keyword detected: {keyword}")
            return True
    print("No matching keyword found.")
    return False

# CNN+Transformer 모델 정의
class CNNTransformer(nn.Module):
    def __init__(self, input_size=257, num_classes=1):
        super(CNNTransformer, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=32, nhead=2)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

# Screaming 여부 판단 함수 (결과 시각화 포함)
def process_audio_file(file_path, sampling_rate, model, threshold=0.5):
    print(f"Loading audio file: {file_path}")
    try:
        audio_data, sr = sf.read(file_path)
        if sr != sampling_rate:
            print(f"Resampling from {sr} to {sampling_rate}")
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sampling_rate)

        audio_data = audio_data / np.max(np.abs(audio_data))
        magnitude = np.abs(stft(audio_data, fs=sampling_rate, nperseg=512, noverlap=256)[2])
        magnitude = magnitude.T
        input_data = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

        with torch.no_grad():
            output = model(input_data)
            predictions = torch.sigmoid(output).numpy()
            print(f"Model predictions (sigmoid values): {predictions}")

        is_screaming = np.any(predictions > threshold)
        print(f"Is screaming detected (any > {threshold}): {is_screaming}")

        # 결과 시각화
        plt.figure(figsize=(12, 8))

        # 오디오 신호 플롯
        plt.subplot(2, 1, 1)
        plt.plot(audio_data[:sampling_rate])
        plt.title("Audio Signal")
        plt.xlim(0, len(audio_data))
        plt.ylim(-1, 1)

        return is_screaming

    except Exception as e:
        print(f"Error during screaming detection: {e}")
        return False

# WAV 파일 재생 함수
def play_siren(wav_file_path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(wav_file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # 재생이 끝날 때까지 대기
    except Exception as e:
        print(f"Error playing siren sound: {e}")

# 메인 함수
if __name__ == "__main__":
    try:
        # 모델 초기화
        model = CNNTransformer(input_size=257, num_classes=1)
        model.load_state_dict(torch.load("cnn_transformer_model.pth"))
        model.eval()

        # 음성 녹음
        recorded_audio_path, sampling_rate = record_audio(duration=2, sampling_rate=44100)

        # Screaming 여부 판단
        s_bool = process_audio_file(recorded_audio_path, sampling_rate, model)
        print(f"s_bool (Screaming Detected): {int(s_bool)}")

        # 도움 요청 키워드 탐지
        transcribed_text = transcribe_audio(recorded_audio_path)
        h_bool = detect_keywords(transcribed_text)
        print(f"h_bool (Help Detected): {int(h_bool)}")

        # 최종 결과 출력
        print(f"s_bool: {int(s_bool)}, h_bool: {int(h_bool)}")

        # 위험상황 출력 및 경고음 재생
        if s_bool or h_bool:
            print("위험상황")
        else:
            print("안전")

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
