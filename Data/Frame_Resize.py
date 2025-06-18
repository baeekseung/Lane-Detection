import cv2
import os
from pathlib import Path

def resize_image(image_path: str, output_path: str, size=(512, 512)):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, size)
    cv2.imwrite(output_path, resized_img)


def resize_all_images(input_dir="frames", output_dir="resized_frames", size=(512, 512)):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(".png"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            resize_image(in_path, out_path, size)

    print(f"[✓] 총 {len(os.listdir(output_dir))}장 리사이즈 완료 → 저장 위치: {output_dir}")


if __name__ == "__main__":
    resize_all_images("frames", "resized_frames")
