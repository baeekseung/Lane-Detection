import json
import os
import numpy as np
import cv2
from pathlib import Path


def labelme_json_to_mask(json_path: str, output_path: str, class_map: dict = {"road": 255}):
    with open(json_path, "r") as f:
        data = json.load(f)

    height = data.get("imageHeight", 512)
    width = data.get("imageWidth", 512)
    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data["shapes"]:
        label = shape["label"].strip().lower()
        if label not in class_map:
            print(f"[!] 알 수 없는 라벨: {label}")
            continue

        if shape.get("shape_type") != "polygon":
            continue

        # float → int 좌표 변환
        points = np.array(shape["points"], dtype=np.float32)
        points = np.round(points).astype(np.int32)

        if len(points) >= 3:
            cv2.fillPoly(mask, [points], color=class_map[label])

    cv2.imwrite(output_path, mask)
    print(f"[✓] 저장됨: {output_path}")


def convert_all_jsons_to_masks(json_dir="resized_frames", output_dir="masks"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            json_path = os.path.join(json_dir, file)
            mask_path = os.path.join(output_dir, file.replace(".json", ".png"))
            labelme_json_to_mask(json_path, mask_path)


if __name__ == "__main__":
    convert_all_jsons_to_masks("resized_frames", "masks")
