import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RoadSegDataset
from unet5 import UNet
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    train_dataset = RoadSegDataset("dataset/images", "dataset/masks")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 200
    save_interval = 50  # 50 에포크마다 저장

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
        
        # 50 에포크마다 모델 저장
        if (epoch + 1) % save_interval == 0:
            save_path = f"unet5_road_seg_{epoch+1}epoch.pth"
            torch.save(model.state_dict(), save_path)
            print(f"중간 모델 저장 완료: {save_path}")

    # 최종 모델 저장
    torch.save(model.state_dict(), "unet5_road_seg_final.pth")
    print("최종 모델 저장 완료: unet_road_seg_final.pth")

if __name__ == "__main__":
    train()
