mport cv2
import numpy as np
import time

# Открываем видеопоток с квадрокоптера
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("/dev/video0") в Linux

# Читаем первый кадр для инициализации
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Инициализация переменных для стабилизации
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, minDistance=30, qualityLevel=0.01)

# Цикл по кадрам видео
while True:
    # Читаем кадр
    ret, frame = cap.read()
    if not ret:
        print("Ошибка! Не удалось считать кадр")
        break

    # Преобразуем кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Вычисляем оптический поток
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    if status is None:
        print("Ошибка! Не удалось определить статус оптического потока")
        break

    # Фильтруем точки, которые не могут быть отслежены
    #если размерности массивов, используемых в индексации с булевыми значениями, не соответствуют.
    #Добавление .ravel() позволяет привести булев массив status к одномерной форме, что должно устранить ошибку.
    good_new = next_pts[status.ravel() == 1]
    good_old = prev_pts[status.ravel() == 1]

    if len(good_new) == 0:
        print("Нет обнаруженных точек")
        continue

    # Вычисляем матрицу трансформации
    M = cv2.estimateAffinePartial2D(good_old, good_new)[0]

    # Применяем матрицу трансформации к кадру
    stable_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    # Отображаем стабилизированный кадр
    cv2.imshow("Stable", stable_frame)

    # Обновляем переменные для следующей итерации
    prev_gray = gray.copy()
    prev_pts = good_new.copy()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Добавляем небольшую паузу между итерациями
    time.sleep(0.1)

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
