import tensorflow as tf

# Загрузите вашу модель (замените на свой фактический путь)
model = tf.keras.models.load_model("disabled_people_model.h5")  # Или другой формат

# Сохраните модель в формате .keras (Keras V3)
model.save('disabled_people_model.keras')

print("Модель успешно преобразована и сохранена в формате .keras")