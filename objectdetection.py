import cv2
import face_recognition
import os
from datetime import datetime
import geocoder
from deepface import DeepFace
import numpy as np
import threading  # To play and stop sound in a separate thread
import pygame  # For sound control

# Initialize pygame mixer for sound control
pygame.mixer.init()

# Initialize face recognition
known_face_encodings = []
known_face_names = []

# Store unknown face encodings to prevent multiple captures
captured_unknown_face_encodings = []

# Directory containing images of known persons
known_faces_dir = r"C:\Users\Acer\Downloads\All_Work\object files\known"

# Directory to save images of unknown persons
unknown_faces_dir = r"C:\Users\Acer\Downloads\All_Work\object files\unknown"

# Load each image file and extract face encodings
for image_name in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, image_name)
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        face_encoding = face_encodings[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(image_name)[0])

# Fetch geolocation once to avoid repeated API calls
g = geocoder.ip('me')

# Set a manual location if geolocation is incorrect
if g.city == "Bhubaneswar" and not g.latlng == [28.7041, 77.1025]:
    location_info = "Location: Bhubaneswar, Odisha, India - Lat: 20.2961, Lng: 85.8245"
else:
    location_info = f"Location: {g.city}, {g.state}, {g.country} - Lat: {g.latlng[0]}, Lng: {g.latlng[1]}"

# Load YOLO object detection model
model_config = "yolov4.cfg"  # Path to the YOLO config file
model_weights = "yolov4.weights"  # Path to the YOLO weights file
coco_names = "coco.names"  # Path to the coco.names file

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(coco_names, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Define dangerous objects that we want to monitor
dangerous_objects = ["knife", "gun", "fire", "scissors"]

# Alarm playing flag
alarm_playing = False

# Function to play alarm sound
def play_alarm():
    global alarm_playing
    sound_file = 'alert.mp3'  # Replace with your sound file (can be .wav or .mp3)
    
    # Load and play sound
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play(-1)  # -1 means the sound will loop indefinitely
    alarm_playing = True

# Function to stop the alarm
def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()  # Stop the alarm sound
        alarm_playing = False
        print("Alarm stopped!")

# Function to start the alarm and stop it after 5 seconds
def start_alarm():
    global alarm_playing
    if not alarm_playing:
        play_alarm()  # Start playing the alarm sound

        # Set a timer to stop the alarm after 5 seconds
        threading.Timer(10.0, stop_alarm).start()

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Perform emotion detection using DeepFace
    try:
        faces = DeepFace.analyze(img_path=rgb_frame, actions=['emotion'], enforce_detection=False)
    except Exception as e:
        print(f"Emotion detection error: {e}")
        faces = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
            # Capture unknown faces but no alarm
            if not any(face_recognition.compare_faces(captured_unknown_face_encodings, face_encoding, tolerance=0.6)):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unknown_image_path = os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg")
                cv2.imwrite(unknown_image_path, face_image)
                captured_unknown_face_encodings.append(face_encoding)

        # Draw a box around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Object Detection with YOLO
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

            # Check if detected object is dangerous
            if classes[class_ids[i]] in dangerous_objects:
                if not alarm_playing:
                    print(f"Dangerous object detected ({classes[class_ids[i]]})! Playing sound alert...")
                    start_alarm()

    # Display emotions if detected
    for face in faces:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, face['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (54, 219, 9), 2)

    # Display the location on the frame
    cv2.putText(frame, location_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (7, 36, 250), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop with 'q' and stop alarm with 's'
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        stop_alarm()

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
