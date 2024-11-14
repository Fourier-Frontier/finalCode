import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

## 데이터 로드 및 균형 조정
class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.chunk_frame = 300
        self.data, self.labels = self.load_data(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_len = self.data[idx].shape[1]
        if seq_len <= self.chunk_frame:
            # 데이터가 너무 짧으면 다시 샘플 선택
            return self.__getitem__(random.randrange(0, len(self.data)))
        str_idx = random.randrange(0, seq_len - self.chunk_frame)
        data = self.data[idx][:, str_idx:str_idx + self.chunk_frame]
        return data, self.labels[idx]

    ## 데이터 로드 및 다운샘플링
    def load_data(self, data_dir):
        data = []
        labels = []

        # 모든 하위 디렉토리에서 .npy 파일 로드
        for file_path in glob.glob(f"{data_dir}/**/*.npy", recursive=True):
            try:
                array = np.load(file_path)
                if array.size == 0:  # 빈 데이터 체크
                    print(f"Empty data in file: {file_path}")
                    continue
                label = 1 if "scream" in file_path else 0
                data.append(array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

        if len(data) == 0:
            raise ValueError("No valid data loaded. Please check the data directory.")

        return data, np.array(labels)

## CNN + Transformer 모델 정의
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
        # x = (batch, freq_bin, seq_len)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)  # (seq_len, batch, feature)
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, feature)
        x = self.fc(x)
        return x.squeeze(1)  # 이진 분류를 위해 출력 차원을 (batch,)로 맞춤

## collate_fn 함수 정의 (패딩 추가)
def collate_fn(batch):
    # 배치에서 가장 긴 시퀀스를 기준으로 패딩
    max_length = max(data.shape[1] for data, _ in batch)
    padded_batch = []
    labels = []

    for data, label in batch:
        pad_size = max_length - data.shape[1]
        if pad_size > 0:
            data = np.pad(data, ((0, 0), (0, pad_size)), mode='constant')
        padded_batch.append(data)
        labels.append(label)

    # 텐서로 변환
    padded_batch = torch.tensor(padded_batch, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    return padded_batch, labels

## 학습 함수
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, labels in dataloader:
            data = data.to(torch.float32)
            labels = labels.to(torch.float32)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.7f}")

## 평가 함수
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(torch.float32)
            labels = labels.to(torch.float32)
            outputs = model(data)
            predictions = torch.sigmoid(outputs) >= 0.5
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

## 데이터 로더 준비
data_dir = '/content/drive/MyDrive/signal&system/dataset/'
batch_size = 32

dataset = AudioDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

## 모델 초기화 및 하이퍼파라미터 설정
input_size = 257
num_classes = 1
num_epochs = 100
learning_rate = 0.001

model = CNNTransformer(input_size=input_size, num_classes=num_classes)
criterion = nn.BCEWithLogitsLoss()  # 이진 분류를 위한 손실 함수
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## 모델 학습
print("Training the model...")
train(model, dataloader, criterion, optimizer, num_epochs)

## 모델 평가
print("Evaluating the model...")
evaluate(model, dataloader)

## 모델 저장
torch.save(model.state_dict(), "cnn_transformer_model.pth")
print("Model saved as cnn_transformer_model.pth")

