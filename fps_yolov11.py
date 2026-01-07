from ultralytics import YOLO
import cv2
import time

# -----------------------
# LOAD MODEL
# -----------------------
model = YOLO("models/best_yolov11.pt")  # Change your model here

# -----------------------
# VIDEO CAPTURE
# -----------------------
cap = cv2.VideoCapture(0)

# Validate camera
if not cap.isOpened():
    print("❌ Error opening the camera")
    exit()

# --------- CONFIGURE RESOLUTION 1280x720 ---------
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# ----------------------------------------------------

prev_time = 0
fps = 0

# ------ Fullscreen Window ------
cv2.namedWindow("FPS Test", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("FPS Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# --------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------
    # MEASURE FPS
    # -----------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # -----------------------
    # YOLO PREDICTION
    # -----------------------
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # -----------------------
    # SHOW FPS ON SCREEN
    # -----------------------
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("FPS Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()