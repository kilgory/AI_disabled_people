import tensorflow as tf
import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.src.optimizers import Adam
from keras.src.regularizers import regularizers
from keras.src.callbacks import EarlyStopping


# 1. Параметры
IMAGE_SIZE = [224, 224]
BATCH_SIZE = 8
EPOCHS = 40

TRAIN_DIR = 'data_sets'


# 2. Загрузка VGG19 без верхних слоев
vgg16 = keras.applications.VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Замораживаем слои VGG19
# print(vgg19.layers[1].get_config())

for layer in vgg16.layers:
    layer.trainable = False

# 3. Добавляем свои слои
x = keras.layers.Flatten()(vgg16.output)
# Начало первого слоя
x = keras.layers.Dense(32,kernel_regularizer=regularizers.L2(0.03))(x)
# x = keras.layers.BatchNormalization()(x)
x= keras.layers.Activation('relu')(x)
# x = keras.layers.Dropout(0.5)(x)

# Начало второго слоя
x = keras.layers.Dense(16,kernel_regularizer=regularizers.L2(0.04))(x)
# x = keras.layers.BatchNormalization()(x)
x= keras.layers.Activation('relu')(x)
# x = keras.layers.Dropout(0.4)(x)
prim = "32 ,kernel_regularizer=regularizers.L2(0.03) + 32,kernel_regularizer=regularizers.L2(0.04) + 8 (0.02)"

# Начало третьего слоя
x = keras.layers.Dense(8,kernel_regularizer=regularizers.L2(0.02))(x)
# x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
# x = keras.layers.Dropout(0.1)(x)

# # Начало четвертого слоя
# x = keras.layers.Dense(64,kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.01))(x)
# x = keras.layers.BatchNormalization()(x)
# x = keras.layers.Activation('relu')(x)
# x = keras.layers.Dropout(0.2)(x)

# # Начало пятого слоя
# x = keras.layers.Dense(16,kernel_regularizer=regularizers.L2(0.01))(x)
# # x = keras.layers.BatchNormalization()(x)
# x = keras.layers.Activation('relu')(x)
# # x = keras.layers.Dropout(0.2)(x)

# Выходной слой
x = keras.layers.Dense(2, activation='softmax')(x)


# 4. Создаем модель
model = keras.Model(inputs=vgg16.input, outputs=x)

# 5. Компилируем модель
# optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# 6. Подготовка данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range = 20,  # Поворачивает изображения случайным образом в диапазоне от -30 до +30 градусов
    # width_shift_range = 0.2,  # Сдвиг по ширине (умеренный)
    # height_shift_range = 0.2,  # Сдвиг по высоте (умеренный)
    shear_range = 0.2,  # Сдвиг (умеренный)
    zoom_range = 0.2,  # Масштабирование (умеренный)
    # horizontal_flip = True,  # Горизонтальное отражение
    fill_mode = 'nearest'  # Заполнение пустых пикселей
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=tuple(IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=tuple(IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 7. Обучение модели

early_stopping = EarlyStopping(
    monitor='val_loss',  # Метрика для мониторинга (валидационная loss)
    patience=5,          # Количество эпох без улучшения, после которых обучение останавливается
    restore_best_weights=True  # Возвращает веса модели к лучшей эпохе
)

history =  model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# plt.text(x=5, y=1, s='Примечание', fontsize=10, color='red') # Здесь x=5 означает 5-я эпоха, а y=1 - значение потери 1
plt.suptitle(prim, fontsize=10, color='red')
plt.show()

model.save('disabled_people_model.h5')

# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()
# print(model.summary())

# 8. Сохранение модели (опционально)
# model.save('smoker_detector_vgg19_simple_sequential.h5')

# print("Обучение завершено. Модель сохранена.")
