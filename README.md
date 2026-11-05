# Распознавание жестов

Программа распознаёт жесты руки через веб-камеру и показывает соответствующие изображения. Для определения положения руки используются MediaPipe и OpenCV, для классификации жестов — модель Random Forest.

## Запуск

1. Установите Python и создайте виртуальное окружение в папке проекта:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install opencv-python mediapipe pandas numpy scikit-learn joblib
   ```

2. В `src/train.py` замените абсолютный путь к `data/data.csv` на путь к файлу на вашем компьютере. В `src/main.py` замените пути к изображениям в словаре `gestures` на пути к файлам из папки `assets`.
3. Из корневой папки проекта запустите:

   ```powershell
   python src/main.py
   ```

При запуске модель обучается на `data/data.csv` и сохраняется в `model.joblib` в корне проекта. Для работы нужна веб-камера; файл `models/hand_landmarker.task` должен находиться на своём месте. Чтобы выйти, нажмите **Q** в окне камеры.
