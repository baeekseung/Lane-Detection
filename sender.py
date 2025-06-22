from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # 0번 웹캠

def generate():
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='172.30.1.85', port=8000)
