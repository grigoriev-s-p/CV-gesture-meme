# Распознавание жестов

Программа распознаёт жесты руки через веб-камеру и показывает соответствующие изображения. Для определения положения руки используются MediaPipe и OpenCV, для классификации жестов — модель Random Forest.

## Запуск

1. Установите Python и создайте виртуальное окружение в папке проекта:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install opencv-python mediapipe pandas numpy scikit-learn joblib
   ```

2. Из корневой папки проекта запустите:

   ```powershell
   python src/main.py
   ```

Для работы нужна веб-камера. Программа загружает модель жестов из `models/model.joblib`, модель MediaPipe из `models/hand_landmarker.task` и изображения из папки `assets`. Чтобы выйти, нажмите **Q** в окне камеры.
