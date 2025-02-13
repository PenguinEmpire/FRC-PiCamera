from flask import Flask, send_file, render_template
import cv2
import numpy as np
import io

app = Flask(__name__)

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Initialize QR code detector
qr_detector = cv2.QRCodeDetector()
# we need to capt
cap = cv2.VideoCapture(0)

@app.route("/api/faces")
def api_faces():
    # Capture image from webcam
    ret, img = cap.read()

    if not ret:
        return "Failed to capture image", 500

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    items = []
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        items.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    return items

@app.route('/api/faces/image', methods=['GET'])
def detect_faces():
    # Capture image from webcam
    ret, img = cap.read()

    if not ret:
        return "Failed to capture image", 500

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Detect QR codes
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(img)
    # Draw QR code bounding boxes
    if retval:
        for i in range(len(decoded_info)):
            if points is not None:
                points = points[i].astype(int)
                for j in range(len(points)):
                    cv2.line(img, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 2)

    # Encode image to PNG
    _, img_encoded = cv2.imencode('.png', img)
    img_bytes = img_encoded.tobytes()

    return send_file(io.BytesIO(img_bytes), mimetype='image/png')

@app.route("/t")
def test():
    return render_template("ui_testing.html")

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        app.logger.info("Releasing webcam.")
        cap.release() # Release the webcam when the app is stopped or crashed
