import cv2
from keras.models import model_from_json
import numpy as np
import os


with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.h5")


haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


labels = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad',}


cat_images = {}
for label in labels.values():
    path = os.path.join("cat_images", f"{label}.jpg")
    if os.path.exists(path):
        cat_images[label] = cv2.imread(path)
    else:
        cat_images[label] = np.zeros((480, 480, 3), dtype=np.uint8)  # fallback blank image


def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)
current_emotion = "neutral"

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)
        pred = model.predict(img)
        current_emotion = labels[pred.argmax()]
        cv2.putText(frame, current_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)


    cat_img = cat_images[current_emotion]
    cat_resized = cv2.resize(cat_img, (frame.shape[1], frame.shape[0]))


    combined = np.hstack((frame, cat_resized))

    cv2.imshow("Emotion + Cat Panel", combined)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

webcam.release()
cv2.destroyAllWindows()
