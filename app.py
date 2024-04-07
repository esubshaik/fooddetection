from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
import numpy
from flask_cors import CORS 

app = Flask(__name__)

# Configure CORS to allow all origins
CORS(app)

# Load YOLOv5 model
model = YOLO("/app/Detect/best.pt")

@app.route('/', methods=["POST"])
def predict():
    if request.method != "POST":
        return jsonify({"error": "Only POST requests are allowed"})

    if "image" not in request.files:
        return jsonify({"error": "No 'image' file part in the request"})

    image_file = request.files["image"]

    try:
        # Use OpenCV to read the image directly from the request file
        img_bytes = image_file.read()
        img = cv2.imdecode(numpy.fromstring(img_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        results = model(img, imgsz=640, conf=0.01)  # includes NMS
        boxes = results
        image_classes = []
        boundings = []
        for i in boxes[0].boxes.numpy().cls:
            image_classes.append(boxes[0].names[i])
        boundings = boxes[0].boxes.numpy().xyxy
        json_object = {class_name: bounding_boxes.tolist() for class_name, bounding_boxes in zip(image_classes, boundings)}
        print(json_object)
        return jsonify({"results": json_object})

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
