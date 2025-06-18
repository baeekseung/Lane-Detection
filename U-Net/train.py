# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RoadSegDataset
from unet import UNet
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    train_dataset = RoadSegDataset("dataset/images", "dataset/masks")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "unet_road_seg.pth")
    print("모델 저장 완료: unet_road_seg.pth")

if __name__ == "__main__":
    train()
