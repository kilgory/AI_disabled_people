import os
from PIL import Image

DATA_DIR = 'data_sets'

def find_palette_image(data_dir):
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                try:
                    img = Image.open(image_path)
                    if img.mode == "P":
                        print(f"Найдено изображение в формате Palette: {image_path}")
                        img.close()
                        # return  # Останавливаемся после первого найденного
                    img.close()
                except Exception as e:
                    print(f"Ошибка при обработке изображения {image_path}: {e}")
    print("Изображения в формате Palette не найдены.")

find_palette_image(DATA_DIR)