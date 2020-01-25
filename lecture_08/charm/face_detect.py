import cv2
import numpy as np

import os

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("Enter your Name : ")

pics = int(input("Enter number of pics : "))

list_faces = []

while True:

    ret, frame = cap.read()

    if ret:

        faces = classifier.detectMultiScale(frame)

        if len(faces) > 0:

            np_faces = np.array(faces)
            best_index = np.product(np_faces[:, 2:], axis=1).argmax()
            x, y, w, h = faces[best_index]
            crop = frame[y:y+h, x:x+w]

            face_img = cv2.resize(crop, (100, 100))

            face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

            face_flat = face_gray.flatten()

            cv2.imshow("face", face_img)

        cv2.imshow("My video", frame)

    key = cv2.waitKey(1)

    if ord("q") == key:
        break

    if ord("c") == key:
        if ret and len(faces) > 0:
            list_faces.append(face_flat)
            print("Captured faces ", len(list_faces))

            if len(list_faces) == pics:
                break



X = np.array(list_faces)

y = np.full((pics, 1), name)

data = np.hstack([y, X])

print(data.shape, data.dtype)

if os.path.exists("faces.npy"):
    old = np.load("faces.npy")
    total = np.vstack([old, data])
    np.save("faces.npy", total)
else:
    np.save("faces.npy", data)

cap.release()
cv2.destroyAllWindows()
