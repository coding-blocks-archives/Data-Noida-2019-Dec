import cv2
import numpy as np

import os
from sklearn.neighbors import KNeighborsClassifier


data = np.load("faces.npy")

X = data[:, 1:].astype(int)

y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, y)

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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

            print(model.predict([face_flat]))

            cv2.rectangle(frame, (x, y), (x+w,y+h), (255, 0, 0), 5)


        cv2.imshow("My video", frame)

    key = cv2.waitKey(1)

    if ord("q") == key:
        break



cap.release()
cv2.destroyAllWindows()
