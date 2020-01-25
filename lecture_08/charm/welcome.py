import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corner = gray[:100, :100]

        cv2.imshow("My video", corner)

        print(frame.shape)

    key = cv2.waitKey(1)

    if ord("q") == key:
        break

    if ord("c") == key:
        cv2.imwrite("profile.jpg", gray)


# cap.release()
# cv2.destroyAllWindows()
