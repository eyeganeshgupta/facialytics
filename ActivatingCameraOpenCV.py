import cv2

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error")
        break

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()

cv2.destroyAllWindows()
