import cv2
import numpy as np
import os


def overlay_mask_on_image(image_path: str, mask_path: str, alpha: float = 0.5):
    """
    단일 이미지와 마스크를 overlay 시각화하는 함수 (반투명 합성)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"마스크를 찾을 수 없습니다: {mask_path}")

    color_mask = np.zeros_like(image)
    color_mask[mask == 255] = (0, 0, 255)  # 빨간색(BGR)으로 칠함

    overlayed = cv2.addWeighted(image, 1.0, color_mask, alpha, 0)
    return overlayed


def show_all_overlays(image_dir="resized_frames", mask_dir="masks"):
    """
    image_dir / mask_dir의 모든 이미지 쌍을 순차적으로 overlay하여 보여줌
    """
    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))

    for file in image_files:
        image_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file)  # 같은 이름이어야 함

        if not os.path.exists(mask_path):
            print(f"[!] 마스크 없음: {mask_path} → 스킵")
            continue

        try:
            overlayed = overlay_mask_on_image(image_path, mask_path)
            cv2.imshow("Overlay", overlayed)
            key = cv2.waitKey(0)  # 아무 키 누르면 다음 이미지로
            if key == 27:  # ESC 키로 종료
                break
        except Exception as e:
            print(f"[X] 오류 발생: {file} → {e}")
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_all_overlays("resized_frames", "masks")
