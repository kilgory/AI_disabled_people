#тут мы просто захватываем кадр.
import cv2

def capture_frame(ip_camera_url):
    """Захватывает кадр из видеопотока IP-камеры."""
    cap = None
    try:
        if cap is None:  # Проверяем, существует ли объект захвата
            cap = cv2.VideoCapture(ip_camera_url)  # Создаем объект захвата
            if not cap.isOpened():
                print(f"Ошибка: Не удалось открыть видеопоток по адресу {ip_camera_url}")
                return None

        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось прочитать кадр из видеопотока.")
            if cap is not None:  # Закрываем только если объект существует
                cap.release()
            return None

        return frame

    except Exception as e:
        print(f"Ошибка при захвате кадра: {e}")
        return None
    finally:
        if cap is not None:
            #cap.release() #Больше не закрываем поток тут!
            pass
