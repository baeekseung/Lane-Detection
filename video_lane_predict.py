import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from U_Net.unet5 import UNet


def load_model(weight_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def preprocess_frame(frame, size=(80, 160)):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
    resized_np = np.array(image.resize((size[1], size[0])))  # (H, W, 3)
    return tensor, resized_np


def postprocess_mask(pred_mask, threshold=0.3):
    pred_mask = pred_mask.squeeze().cpu().numpy()  # (H, W)
    bin_mask = (pred_mask > threshold).astype(np.uint8) * 255
    return bin_mask


def overlay_mask(image_np, mask_np, alpha=0.9):
    color_mask = np.zeros_like(image_np)
    color_mask[mask_np == 255] = (0, 0, 255)  # 빨간색
    blended = cv2.addWeighted(image_np, 1.0, color_mask, alpha, 0)
    return blended


def process_video(video_path, output_path="output_prediction_unet5.mp4", model_path="./U_Net/unet5_road_seg_final.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"[X] 영상 파일 열기 실패: {video_path}")

    width = 160
    height = 80
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[✓] 영상 처리 시작: {video_path}")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor, resized_np = preprocess_frame(frame, size=(height, width))

        with torch.no_grad():
            output = model(input_tensor.to(device))
            output = torch.sigmoid(output)

        mask_np = postprocess_mask(output)
        overlay = overlay_mask(resized_np, mask_np)

        out.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  # RGB → BGR

        frame_count += 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count % 5000 == 0:
            print(f"  ▶ {frame_count} / {total_frames} 프레임 처리 완료")

    cap.release()
    out.release()
    print(f"[✓] 완료: 결과 저장됨 → {output_path}")


if __name__ == "__main__":
    input_video = "./Data/videos/Test_video.mp4"
    process_video(input_video)
