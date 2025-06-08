import tensorflow as tf

model = tf.keras.models.load_model("disabled_people_model.h5")  


model.save('disabled_people_model.keras')

print("Модель успешно преобразована и сохранена в формате .keras")
