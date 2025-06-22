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

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/baeekseung/Real-Time_Lane-Detection.git
cd Real-Time_Lane-Detection
```
### 2. Install dependencies
> [!Note]
> `torch` and `torchvision` are intentionally excluded from `requirements.txt`.
> Please install them manually based on your system environment. We recommend installing the **GPU version** of PyTorch for better performance.
```bash
pip install -r requirements.txt
```
### 3. Model download
You can download the pre-trained U-Net model from the following link:  
**[Download UNet Model (Google Drive)](https://drive.google.com/file/d/1rROuXDTVJ7fodC_6Lskzqee2alBz-iIM/view?usp=sharing)**  
  
After downloading, place the model file (e.g., unet_road_seg.pth) in the following path:  
👉 **Real-Time_Lane-Detection/weights/unet_road_seg.pth**

### 4. Test
To verify that the installation was successful, run the following command:  
```bash
python predict.py
```
If everything is set up correctly, this will load the model and perform lane segmentation on a sample image.  

## Run real-time lane detection on live webcam stream
> [!Note]
> The sender pc and receiver pc must be connected to the same local network (LAN or Wi-Fi) for the video stream to work properly.

### 1. Sender.py
Before running the sender, make sure:

- A webcam is connected and accessible (e.g., built-in or USB camera).
- The server PC is on the **same local network (LAN or Wi-Fi)** as the client.
- You know the server PC's **local IP address** (e.g., `192.168.x.x` or `10.x.x.x`).
```bash
python stream_server.py --host <YOUR_IP_ADDRESS> --port <PORT>
# --host is the IP address of the PC running the server
# port is the port number to serve the video (default is 8000)
# Example : python stream_server.py --host 192.168.0.101 --port 8000
```
### 2. receiver.py
To start the client that receives video and performs lane segmentation:  
```bash
python client_predict_stream.py --ip <SENDER_PC_IP> --port <PORT>
# ip: IP address of the PC running stream_server.py
# port: Port number used by the sender (default is 8000)
# Example : python client_predict_stream.py --ip 192.168.0.101 --port 8000
```




