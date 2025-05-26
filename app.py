from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from ultralytics import YOLO
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ANNOTATED_FOLDER = 'static/annotated'
VIDEO_FOLDER = 'static/videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

model = YOLO('yolov8n.pt')

def detect_objects_image(input_path, output_path):
    image = cv2.imread(input_path)
    results = model(image)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = f"{model.names[cls]}"
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(output_path, image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    file = request.files['image']
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['ANNOTATED_FOLDER'], filename)
    file.save(input_path)
    detect_objects_image(input_path, output_path)
    return render_template('result.html', image_path=output_path)

@app.route('/upload-video', methods=['POST'])
def upload_video():
    file = request.files['video']
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['ANNOTATED_FOLDER'], filename)
    file.save(input_path)
    return render_template('result.html', image_path=input_path)

@app.route('/live')
def live_detection():
    return "Live webcam detection is not available in hosted environments like Render."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
