import cv2
import numpy as np

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is None:
        print("No face detected in frame")
        return None

    detected_faces = []
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]
        detected_faces.append(cropped_face)
    return detected_faces

img = cv2.imread("./men.png",1)

cropped_faces = face_extractor(img)
print(len(cropped_faces))

