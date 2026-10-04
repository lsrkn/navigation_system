import os
import sys
import cv2
import numpy as np
import pickle
from pathlib import Path
import logging

# Логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

try:
    from main import load_model, predict_position, load_embeddings, MahalanobisDistance
    logger.info("Функции из main.py успешно импортированы.")
except ImportError as e:
    logger.error(f"Не удалось импортировать функции из main.py: {e}")
    logger.info("Убедитесь, что вы запускаете скрипт из корневой директории проекта или добавили её в sys.path.")
    logger.info("Пример: cd /path/to/navigation_system && python testing/test_maps/test_coordinates_from_images.py")
    sys.exit(1)

MODEL_ENCODER_PATH = project_root / "models" / "best_encoder_02_08.pth"  # Путь к энкодеру
MODEL_AUTOENCODER_PATH = project_root / "models" / "autoencoder_model.h5"  # Путь к автоэнкодеру
EMBEDDINGS_PATH = project_root / "embeddings.pkl"  # Путь к файлу с эмбеддингами
INV_COV_MATRIX_PATH = project_root / "inv_cov_matrix.npy"  # Путь к инверсной ковариационной матрице
TEST_IMAGES_DIR = Path(__file__).resolve().parent  # Директория с тестовыми изображениями (testing/test_maps)
RESULTS_DIR = TEST_IMAGES_DIR / "results"  # Директория для сохранения результатов

def create_results_dir():
    RESULTS_DIR.mkdir(exist_ok=True)
    logger.info(f"Директория для результатов создана: {RESULTS_DIR}")

def load_pretrained_assets():
    logger.info("Загрузка обученных моделей и данных...")
    try:
        # Загрузка модели энкодера
        encoder = load_model(model_path=str(MODEL_ENCODER_PATH), model_type='encoder')
        logger.info("Энкодер загружен.")

        # Загрузка эмбеддингов
        embeddings = load_embeddings(path=str(EMBEDDINGS_PATH))
        logger.info(f"Эмбеддинги загружены. Количество тайлов: {len(embeddings['coordinates'])}")

        # Загрузка инверсной ковариационной матрицы
        inv_cov_matrix = np.load(str(INV_COV_MATRIX_PATH))
        logger.info("Инверсная ковариационная матрица загружена.")

        return encoder, embeddings, inv_cov_matrix
    except FileNotFoundError as e:
        logger.error(f"Ошибка: не найден файл. {e}")
        logger.error(f"Убедись, что файлы {MODEL_ENCODER_PATH}, {EMBEDDINGS_PATH}, {INV_COV_MATRIX_PATH} существуют.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)

def preprocess_image(image_path, target_size=(64, 64)):
    
    # Чтение изображения
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Не удалось прочитать изображение: {image_path}")
        return None

    # Изменение размера
    image = cv2.resize(image, target_size)
    # Нормализация (пример, как в Keras/PyTorch: /255.0)
    image = image.astype('float32') / 255.0
    # image = np.transpose(image, (2, 0, 1))
    return image

def annotate_image_with_position(image_path, predicted_position, output_path):
    
    # Загрузка исходной карты
    map_path = project_root / "map" / "test_map_crop.png"
    if not map_path.exists():
        logger.warning(f"Карта не найдена по пути {map_path}. Аннотация не будет создана.")
        return False

    map_image = cv2.imread(str(map_path))
    if map_image is None:
        logger.warning(f"Не удалось прочитать карту: {map_path}")
        return False

    # Проверка корректности координат
    map_height, map_width = map_image.shape[:2]
    x, y = predicted_position
    if 0 <= x < map_width and 0 <= y < map_height:
        # Отрисовка круга и координат
        cv2.circle(map_image, (x, y), radius=10, color=(0, 255, 0), thickness=2)
        cv2.putText(map_image, f"({x}, {y})", (x+15, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # Сохранение аннотированной карты
        cv2.imwrite(str(output_path), map_image)
        logger.info(f"Аннотированная карта сохранена: {output_path}")
        return True
    else:
        logger.warning(f"Предсказанные координаты ({x}, {y}) вне границ карты ({map_width}x{map_height}).")
        return False

def main():
    
    logger.info("=== Тестирование навигационной системы: определение координат по изображениям ===")

    # 1. Создание директории для результатов
    create_results_dir()

    # 2. Загрузка всех необходимых обученных файлов
    encoder, embeddings, inv_cov_matrix = load_pretrained_assets()

    # 3. Получение списка изображений из тестовой директории
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in TEST_IMAGES_DIR.iterdir() if f.suffix.lower() in image_extensions]
    if not image_files:
        logger.warning(f"В директории {TEST_IMAGES_DIR} нет изображений с расширениями {image_extensions}.")
        return
    logger.info(f"Найдено изображений для тестирования: {len(image_files)}")

    # 4. Обработка каждого изображения
    results_summary = []
    for img_path in image_files:
        logger.info(f"\n--- Обработка изображения: {img_path.name} ---")
        # 4.1. Предобработка изображения
        processed_img = preprocess_image(img_path)
        if processed_img is None:
            continue
        # 4.2. Предсказание позиции (координат)
        # Функция predict_position должна использовать encoder, embeddings и inv_cov_matrix
        try:
            predicted_position = predict_position(processed_img, encoder, embeddings, inv_cov_matrix)
            logger.info(f"Предсказанные координаты: {predicted_position}")
            results_summary.append((img_path.name, predicted_position))
        except Exception as e:
            logger.error(f"Ошибка при предсказании позиции для {img_path.name}: {e}")
            continue
        # 4.3. Аннотирование изображения и сохранение результата
        output_img_path = RESULTS_DIR / f"annotated_{img_path.stem}.png"
        if annotate_image_with_position(img_path, predicted_position, output_img_path):
            logger.info(f"Результат сохранен в {output_img_path}")

    # 5. Вывод сводки результатов
    logger.info("\n=== Сводка результатов тестирования ===")
    logger.info(f"{'Изображение':<30} | {'Предсказанные координаты (X, Y)'}")
    logger.info("-" * 70)
    for img_name, position in results_summary:
        logger.info(f"{img_name:<30} | {position}")
    logger.info(f"\nТест завершен. Аннотированные изображения и сводка сохранены в {RESULTS_DIR}.")

if __name__ == "__main__":
    main()
