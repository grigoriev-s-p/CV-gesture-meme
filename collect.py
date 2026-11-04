import cv2
import mediapipe as mp
import time
import csv



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

record = False
gesture = "Unknown"

last_saved = 0
save_interval = 100

counts = {"palm":0,"fist":0,"like":0, "Unknown":0}
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
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        record = not record
        cv2.putText(img, str(record), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if key == ord('1'):
        gesture = "palm"
        record = False
    if key == ord('2'):
        gesture = "fist"
        record = False
    if key == ord('3'):
        gesture = "like"
        record = False

    if result.hand_landmarks:

        for points in result.hand_landmarks:
            mpDraw.draw_landmarks(img, points, connections)
            landmarks = [coord for point in points for coord in (point.x, point.y, point.z)]
            if record and gesture!= "Unknown" and timestamp - last_saved >= save_interval:
                with open("test.csv", "a", newline="") as file:
                    writer = csv.writer (file)
                    writer.writerow(landmarks + [gesture])
                    counts[gesture]+=1
                last_saved = timestamp
            cv2.putText(img, gesture, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(counts[gesture]), (330, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Result:", img)

    if key == ord('q'):
        break
camera.release()
hands.close()
cv2.destroyAllWindows()