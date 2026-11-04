import cv2
import mediapipe as mp
import time
import joblib
import pandas as pd
from train import normalize

GESTURE_DELAY = 0.4
WINDOW_SIZE = 800


class ImageWindow:
    def __init__(self):
        self.current_path = None
        self.pending_path = None
        self.gesture_started = 0.0

    def update_gesture(self, path):
        if path is None:
            self.pending_path = None
            self.hide()
            return

        now = time.monotonic()
        if path != self.pending_path:
            self.pending_path = path
            self.gesture_started = now
            self.hide()
        elif now - self.gesture_started >= GESTURE_DELAY:
            self.show(path)

    def show(self, path):
        if path == self.current_path:
            return

        img = cv2.imread(path)
        img = cv2.resize(img, (WINDOW_SIZE, WINDOW_SIZE))
        cv2.namedWindow("Meme", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Meme", img)
        self.current_path = path

    def hide(self):
        if self.current_path is not None:
            cv2.destroyWindow("Meme")
            self.current_path = None

    def close(self):
        self.hide()


model = joblib.load("model.joblib")

camera = cv2.VideoCapture(0)
mpHands = mp.tasks.vision.HandLandmarker
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)


hands = mpHands.create_from_options(options)
mpDraw = mp.tasks.vision.drawing_utils


connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

gestures = {
    gesture: f"Images/{gesture}.png"
    for gesture in ("fist", "fuck", "idk", "peace", "rock", "palm", "like")
}

img_window = ImageWindow()
while True:
    success, img = camera.read()
    if not success:
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mpImage = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )
    timestamp = time.monotonic_ns()//1_000_000
    result = hands.detect_for_video(mpImage, timestamp)

    gesture = None
    if result.hand_landmarks:
        for points in result.hand_landmarks:
            mpDraw.draw_landmarks(img, points, connections)
            landmarks = [coord for point in points for coord in (point.x, point.y, point.z)]
            sample = normalize(pd.DataFrame([landmarks + ["Unknown"]]))
            gesture = model.predict(sample.iloc[:,:63])[0]

    img_window.update_gesture(gestures.get(gesture))

    cv2.imshow("Result:", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
hands.close()
img_window.close()
cv2.destroyAllWindows()
