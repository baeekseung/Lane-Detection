# Real-Time driving road segmentation with U-Net

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge)
![Pillow](https://img.shields.io/badge/Pillow-CC66CC?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

## Introduction
This project presents a real-time lane segmentation system built using a U-Net-based deep learning model.  
We sampled frames from YouTube driving videos and manually labeled the road areas in each frame. This process resulted in a training dataset of approximately 11,200 image–mask pairs for road segmentation. U-Net architectures were then trained on this dataset.  
The system supports real-time road segmentation by applying a trained U-Net model to incoming video streams. It can receive live driving footage from a remote webcam via a lightweight streaming server, perform lane segmentation frame-by-frame, and display the results with minimal latency. The model's predictions are overlaid on the original frames, and the real-time FPS is displayed to monitor performance.

## Installation & Usage
### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-project-name.git
cd your-project-name
```





> [!Note]
> Table의 pre-training, fine-tuning은 본 프로젝트에서 실행한 학습을 의미합니다.  
> Base model들은 이미 이전에 pre-training/fine-tuning을 거친 모델들로 본 프로젝트에서는 transfer learning 하였습니다.
