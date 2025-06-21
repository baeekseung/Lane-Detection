import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from unet5 import UNet
from PIL import Image
import os


def load_model(weight_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path, size=(80, 160)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
    # numpy 변환 시 (height, width, 3)로 맞추기 위해 size를 뒤집어서 전달
    original_np = np.array(image.resize((size[1], size[0])))
    return tensor, original_np  # tensor, original image (numpy)


def postprocess_mask(pred_mask, threshold=0.3):
    pred_mask = pred_mask.squeeze().cpu().numpy()  # (H, W)
    # 배열을 문자열로 변환
    arr_str = np.array2string(pred_mask, separator=', ', threshold=np.inf)
    bin_mask = (pred_mask > threshold).astype(np.uint8) * 255  # 0 or 255
    return bin_mask


def overlay_mask(image_np, mask_np, alpha=0.9):
    # image_np와 mask_np의 shape이 항상 (80, 160)으로 일치한다고 가정
    color_mask = np.zeros_like(image_np)
    color_mask[mask_np == 255] = (0, 0, 255)  # 빨간색 (BGR)
    blended = cv2.addWeighted(image_np, 1.0, color_mask, alpha, 0)
    return blended


def predict(image_path, model_path="unet5_road_seg_final.pth", output_path="prediction_overlay.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 로드
    model = load_model(model_path, device)

    # 2. 입력 이미지 전처리
    input_tensor, original_np = preprocess_image(image_path)

    # 3. 추론
    with torch.no_grad():
        output = model(input_tensor.to(device))
        output = torch.sigmoid(output)

    # 4. 마스크 후처리
    mask_np = postprocess_mask(output)

    # 5. 시각화 및 저장
    overlay = overlay_mask(original_np, mask_np)
    cv2.imwrite(output_path, overlay)
    print(f"[✓] 추론 결과 저장됨 → {output_path}")

    

    # 선택적으로 바로 보기
    cv2.imshow("Prediction", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    


if __name__ == "__main__":
    test_image = "../Data/Frames/frame_249135.png"  # 추론할 이미지 경로
    predict(test_image)
