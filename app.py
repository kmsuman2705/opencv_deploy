from flask import Flask, Response, render_template, request, redirect, url_for, flash
import cv2
import face_recognition
import os
from datetime import datetime
import geocoder
from deepface import DeepFace
import numpy as np
import logging
# Initialize Flask
app = Flask(__name__)
app.secret_key = 'your_generated_secret_key'  # Update this with a generated key

from flask_cors import CORS
CORS(app)

logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')

# Initialize face recognition
known_face_encodings = []
known_face_names = []
captured_unknown_face_encodings = []

# Directory containing images of known persons
known_faces_dir = r"/home/ec2-user/known"

# Directory to save images of unknown persons
unknown_faces_dir = r"/home/ec2-user/unknown"

# Load each image file and extract face encodings
for image_name in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, image_name)
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        face_encoding = face_encodings[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(image_name)[0])

# Fetch geolocation
g = geocoder.ip('me')
if g.city == "Bhubaneswar" and not g.latlng == [28.7041, 77.1025]:
    location_info = "Location: Bhubaneswar, Odisha, India - Lat: 20.2961, Lng: 85.8245"
else:
    location_info = f"Location: {g.city}, {g.state}, {g.country} - Lat: {g.latlng[0]}, Lng: {g.latlng[1]}"

# Load YOLO object detection model
model_config = "yolov4.cfg"
model_weights = "yolov4.weights"
coco_names = "coco.names"

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
layer_names = net.getLayerNames()

# Handle different versions of OpenCV
out_layer_indices = net.getUnconnectedOutLayers()
if len(out_layer_indices.shape) == 1:
    output_layers = [layer_names[i - 1] for i in out_layer_indices]
else:
    output_layers = [layer_names[i[0] - 1] for i in out_layer_indices]

with open(coco_names, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

dangerous_objects = ["knife", "gun", "fire", "scissors"]

# Initialize webcam
video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        print(f"Detected face locations: {face_locations}")
        print(f"Face encodings: {face_encodings}")

        try:
            faces = DeepFace.analyze(img_path=rgb_frame, actions=['emotion'], enforce_detection=False)
        except Exception as e:
            print(f"Error in DeepFace analysis: {e}")
            faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin() if face_distances.size > 0 else -1

            if matches[best_match_index] if best_match_index >= 0 else False:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
                if not any(face_recognition.compare_faces(captured_unknown_face_encodings, face_encoding, tolerance=0.6)):
                    top, right, bottom, left = face_location
                    face_image = frame[top:bottom, left:right]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unknown_image_path = os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg")
                    cv2.imwrite(unknown_image_path, face_image)
                    captured_unknown_face_encodings.append(face_encoding)

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Object detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x, center_y, width, height = (obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for face in faces:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, face['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (54, 219, 9), 2)

        cv2.putText(frame, location_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (7, 36, 250), 2, cv2.LINE_AA)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define allowed file types for uploads
app.config['UPLOAD_FOLDER'] = 'known'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


@app.route("/show_log")
def show_log():
    try:
        with open('app.log', 'r') as log_file:
            log_content = log_file.read()
        return f"<pre>{log_content}</pre>"
    except Exception as e:
        return f"Error reading log file: {str(e)}"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/member')
def member():
    # List all known member images
    known_images = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    return render_template('member.html', known_images=known_images)

@app.route('/add_member', methods=['POST'])
def add_member():
    if 'member_image' not in request.files:
        flash('No file part.', 'error')
        return redirect(url_for('member'))

    file = request.files['member_image']
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            face_encoding = face_encodings[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])
            flash('Member added successfully!', 'success')
        else:
            flash('No face detected in the image.', 'error')
    else:
        flash('Invalid file type.', 'error')

    return redirect(url_for('member'))

@app.route('/email', methods=['GET', 'POST'])
def email():
    if request.method == 'POST':
        email = request.form.get('email')
        if email:
            # Save the email to a file or database (for simplicity, we're using a file)
            with open('emails.txt', 'a') as f:
                f.write(email + '\n')
            flash('Email added successfully!', 'success')
        else:
            flash('Email cannot be empty.', 'error')

    # Read the emails from the file
    try:
        with open('emails.txt', 'r') as f:
            emails = f.read().splitlines()
    except FileNotFoundError:
        emails = []

    return render_template('email.html', emails=emails)

@app.route('/delete_email/<email>', methods=['POST'])
def delete_email(email):
    try:
        with open('emails.txt', 'r') as f:
            emails = f.read().splitlines()
        emails.remove(email)
        with open('emails.txt', 'w') as f:
            f.write('\n'.join(emails))
        flash('Email deleted successfully!', 'success')
    except Exception as e:
        flash('An error occurred while deleting the email.', 'error')

    return redirect(url_for('email'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
from flask_socketio import SocketIO
socketio = SocketIO(app)
if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, ssl_context=('cert.pem', 'key.pem'))
