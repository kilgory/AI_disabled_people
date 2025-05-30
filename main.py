import cv2
import os
import time
from dotenv import load_dotenv
import tensorflow as tf
import cams  # Импортируем модуль cams

# --- Загрузка переменных окружения ---
load_dotenv()

IP_CAMERA_URL = os.getenv("IP_CAMERA_URL")
MODEL_PATH = "MODEL_PATH"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CONFIDENCE_THRESHOLD = 0.7  # Минимальная уверенность для обнаружения
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

if not IP_CAMERA_URL:
    raise ValueError("Необходимо задать IP_CAMERA_URL в .env файле")
if not MODEL_PATH:
     raise ValueError("Необходимо задать MODEL_PATH в .env файле")
if not OUTPUT_DIR:
    raise ValueError("Необходимо задать OUTPUT_DIR в .env файле")

# Создаем папку, если ее нет ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Загрузка модели ---
model = tf.keras.models.load_model("disabled_people_model.keras")

# --- Функция для предобработки изображения ---
def preprocess_image(frame):
    """Изменяет размер и нормализует изображение."""
    try:
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = tf.keras.utils.img_to_array(resized_frame)
        img_array = tf.expand_dims(img_array, axis=0) # Добавляем размерность батча
        return img_array / 255.0  # Нормализация
    except Exception as e:
        print(f"Ошибка при предобработке изображения: {e}")
        return None

# --- Функция для предсказания ---
def predict(model, img_array):
    """Выполняет предсказание на изображении."""
    try:
        predictions = model.predict(img_array)
        return predictions
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return None


# --- Основной код ---
image_counter = 0 # Инициализируем счетчик изображений

while True:
    try:
        # --- Захватываем кадр с камеры (используем модуль cams) ---
        frame = cams.capture_frame(IP_CAMERA_URL)

        if frame is not None:
            processed_frame = preprocess_image(frame)

            if processed_frame is not None:
                predictions = predict(model, processed_frame)

                # --- Обработка результата
                if predictions[0][0] > CONFIDENCE_THRESHOLD:
                    print("Обнаружен инвалид!", " Точность:", round(predictions[0][0]*100,2),"%")
                    # Сохраняем кадры
                    image_name = f"frame_{image_counter:04d}.jpg"  # Формируем имя файла
                    image_path = os.path.join(OUTPUT_DIR, image_name)  # Полный путь к файлу
                    cv2.imwrite(image_path, frame)  # Сохраняем как JPEG
                    print(f"Кадр успешно сохранен в {image_path}")
                    image_counter += 1  # Увеличиваем счетчик

                else:
                    # print("Инвалид не обнаружен.")
                    pass


            else:
                print("Ошибка: Не удалось обработать кадр.")
        else:
            print("Ошибка: Не удалось захватить кадр.")

    except Exception as e:
        print(f"Общая ошибка: {e}")

    time.sleep(5) # Ждем 5 секунды