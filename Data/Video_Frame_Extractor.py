import cv2
import os
import random

# mp4 파일 경로와 저장 폴더, 추출할 프레임 개수 k를 지정하세요.
VIDEO_PATH = "videos/KOREA DRIVE ｜ 4K Highway Driving Autumn 2022 ｜ 영동고속도로 전구간 주행영상 백색소음 ASMR.mp4"
OUTPUT_DIR = "frames"
k = 1000  # 원하는 프레임 개수로 변경하세요

# 저장 폴더가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():d
    print("비디오 파일을 열 수 없습니다.")
    exit()

# 전체 프레임 수 구하기
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"총 프레임 수 : {frame_count}")

# k개의 랜덤 프레임 인덱스 선택 (중복 없음)
random_indices = sorted(random.sample(range(frame_count), k))

saved = 0
for idx in random_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        filename = os.path.join(OUTPUT_DIR, f"frame_{idx:06d}.png")
        cv2.imwrite(filename, frame)
        saved += 1
        if saved%100==0:
            print(f"프레임 {idx} 저장 완료: {filename}")
    else:
        print(f"프레임 {idx}를 읽을 수 없습니다.")

cap.release()
print(f"총 {saved}개의 프레임이 저장되었습니다.")