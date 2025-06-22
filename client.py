# client_predict_stream.py (수신자 PC)
import cv2
import numpy as np
import torch
from U_Net.unet3 import UNet
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import time

server_ip = "172.30.1.85"
port_num = "8000"

# Load Modelc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("./U_Net/models/unet3_road_seg.pth", map_location=device))
model.eval()

# Inference Helpers
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor()
])

def predict_frame(frame_np):
    image = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    mask = (output > 0.4).astype(np.uint8) * 255
    return mask

def overlay_mask(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[mask == 255] = (0, 0, 255)
    return cv2.addWeighted(image, 1.0, color_mask, 0.7, 0)

# Stream Video from Sender
url = f'http://{server_ip}:{port_num}/video'
stream = requests.get(url, stream=True)

prev_time = time.time()
bytes_buffer = b''
for chunk in stream.iter_content(chunk_size=1024):
    bytes_buffer += chunk
    a = bytes_buffer.find(b'\xff\xd8')  # JPEG 시작
    b = bytes_buffer.find(b'\xff\xd9')  # JPEG 끝
    if a != -1 and b != -1:
        jpg = bytes_buffer[a:b+2]
        bytes_buffer = bytes_buffer[b+2:]

        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
    else:
        continue

    current_time = time.time()
    fps = 1 / (current_time - prev_time+0.000001)
    prev_time = current_time

    resized_frame = cv2.resize(frame, (160, 80))
    mask = predict_frame(frame)
    overlay = overlay_mask(resized_frame, mask)

    display_overlay = cv2.resize(overlay, (640, 320))

    cv2.putText(display_overlay, f"FPS: {fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Lane Detection", display_overlay)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
