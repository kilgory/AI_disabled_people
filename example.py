import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. Загрузка модели
model = load_model('disabled_people_model.h5')
class_names = ['Инвалид', 'Не инвалид']
img_width, img_height = 224, 224


# 4. Путь к изображению,
image_path = 'tests/2.png'

# 5. Функция для загрузки и предобработки изображения
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для batch_size
    img_array = img_array / 255.0  # Нормализация (если вы делали это при обучении)
    return img_array

# 6. Загрузка и предобработка изображения
img_array = load_and_preprocess_image(image_path)

# 7. Предсказание
predictions = model.predict(img_array)
probabilities = predictions[0]  # Извлекаем вероятности для одного изображения
print(predictions)

# 8. Получение предсказанного класса и его вероятности
predicted_class_index = np.argmax(probabilities)
predicted_class = class_names[predicted_class_index]
confidence = probabilities[predicted_class_index]

# 9. Вывод результатов
print(f"Предсказанный класс: {predicted_class}")
print(f"Уверенность: {confidence:.4f}")