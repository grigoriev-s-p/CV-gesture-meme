import cv2
import mediapipe as mp
import time
import joblib
import pandas as pd
import tkinter as tk
from src.train import normalize

class ImageWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.label = tk.Label(self.root)
        self.label.pack()
        self.current_path = None

    def show(self, path):
        if path == self.current_path:
            return

        self.img = tk.PhotoImage(file=path)
        self.label.config(image=self.img)
        self.current_path = path
    def close(self):
        self.root.destroy()


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

gestures = {"fist":"C:/Users/11211/Desktop/github/pet-projec/Images/fist.png",
            "fuck":"C:/Users/11211/Desktop/github/pet-projec/Images/fuck.png",
            "idk":"C:/Users/11211/Desktop/github/pet-projec/Images/idk.png",
            "peace":"C:/Users/11211/Desktop/github/pet-projec/Images/peace.png",
            "rock": "C:/Users/11211/Desktop/github/pet-projec/Images/rock.png",
            "palm": "C:/Users/11211/Desktop/github/pet-projec/Images/palm.png",
            "like": "C:/Users/11211/Desktop/github/pet-projec/Images/like.png"}

img_window = ImageWindow()
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
            img_window.show(gestures[gesture])

    cv2.imshow("Result:", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    img_window.root.update()