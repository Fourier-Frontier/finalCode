import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy
import serial
from torch import nn
from scipy.signal import stft

# 모델 정의 (CNN+Transformer)
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

# 모델 불러오기
model = CNNTransformer(input_size=257, num_classes=1)
model.load_state_dict(torch.load("cnn_transformer_model.pth"))
model.eval()

# Arduino 연결 설정 (COM 포트는 환경에 맞게 설정)
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)

# 실시간 음성 녹음 설정
duration = 2  # 녹음 시간 (초)
sampling_rate = 16000

# Fourier Transform 함수
def perform_fourier_transform(audio_data):
    _, _, freq_data = stft(audio_data, fs=sampling_rate, window='rectangular', nperseg=512, noverlap=0, nfft=512)
    magnitude = np.abs(freq_data)
    return magnitude

# 실시간 예측 및 시각화 함수
def real_time_prediction():
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    audio_line, = ax[0].plot([], [], lw=2)
    prediction_text = ax[1].text(0.5, 0.5, "", fontsize=20, ha='center')

    while True:
        print("Recording...")
        audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
        sd.wait()
        audio_data = audio_data.flatten()

        # Fourier Transform 및 모델 예측
        magnitude = perform_fourier_transform(audio_data)
        input_data = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(input_data)
            prediction = torch.sigmoid(output).item()

        # 예측 결과 시각화 업데이트
        ax[0].cla()
        ax[0].plot(audio_data[:sampling_rate * duration])
        ax[0].set_title("Real-Time Audio Signal")
        ax[0].set_xlim(0, len(audio_data))
        ax[0].set_ylim(-1, 1)

        ax[1].cla()
        result_text = "Scream Detected!" if prediction > 0.5 else "No Scream"
        ax[1].text(0.5, 0.5, result_text, fontsize=20, ha='center', color='red' if prediction > 0.5 else 'blue')
        ax[1].set_title("Prediction Result")

        plt.pause(0.1)

        # Arduino로 예측 결과 전송
        if prediction > 0.5:
            arduino.write(b'1')  # Scream 감지 시 '1' 전송
        else:
            arduino.write(b'0')  # 감지 안 됨 시 '0' 전송

        # 종료 조건
        user_input = input("Type 'stop' to end the program: ")
        if user_input.lower() == 'stop':
            break

    plt.ioff()
    plt.show()
    arduino.close()

# 메인 함수 실행
if __name__ == "__main__":
    real_time_prediction()

