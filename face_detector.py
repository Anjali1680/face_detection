import cv2

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError(f"Failed to load cascade at {cascade_path}")

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    raise IOError("Cannot open webcam. Check camera permissions / index.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detectMultiScale parameters: scaleFactor, minNeighbors, minSize (optional)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Detector", frame)

        # Press 'q' to quit, or ESC (27)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
