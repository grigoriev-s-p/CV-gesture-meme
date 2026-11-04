import cv2
import mediapipe as mp
import time
import joblib
import pandas as pd
from train import normalize


model = joblib.load("model.joblib")

camera = cv2.VideoCapture(0)
mpHands = mp.tasks.vision.HandLandmarker
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)


hands = mpHands.create_from_options(options)
mpDraw = mp.tasks.vision.drawing_utils


connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

while True:
    success, img = camera.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mpImage = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )
    timestamp = time.monotonic_ns()//1_000_000
    result = hands.detect_for_video(mpImage, timestamp)

    if result.hand_landmarks:
        for points in result.hand_landmarks:
            mpDraw.draw_landmarks(img, points, connections)
            landmarks = [coord for point in points for coord in (point.x, point.y, point.z)]
            sample = normalize(pd.DataFrame([landmarks + ["Unknown"]]))
            gesture = model.predict(sample.iloc[:,:63])[0]
            cv2.putText(
                img,
                gesture,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Result:", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
