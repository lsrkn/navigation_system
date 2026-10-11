from picamera2 import Picamera2
from datetime import datetime
import time
import os
import numpy as np
from PIL import Image
from scipy.spatial import distance
import numpy as np 
import pandas as pd 
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import pickle



# train_on_gpu = torch.cuda.is_available()

# if not train_on_gpu:
#     print('CUDA is not available')
# else:
#     print('CUDA is available!')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Constants
TILE_STEP = 50  # Шаг для генерации тайлов карты (используется в 1_prepare_dataset.ipynb для создания датасета)
TILE_SIZE = 300  # Размер тайла (изображения) для обучения и инференса (300x300 пикселей)
ENCODED_SPACE_DIM = 256  # Размерность латентного пространства (эмбеддингов) автоэнкодера
NEARBY_DELTA = 300  # Радиус для поиска ближайших эмбеддингов (порог для фильтрации по расстоянию)
MAP_PATH = './map/test_map_crop.png'  # Путь к изображению карты
ENCODER_MODEL_PATH = './models/best_encoder_02_08.pth'  # Путь к обученной модели энкодера
EMBEDDINGS_PATH = "embeddings.pkl"  # Путь к файлу с сохраненными эмбеддингами (создается в 3_prepare_emb_matx_test.ipynb)
MATRIX_PATH = "inv_cov_matrix.npy"  # Путь к файлу с инверсной ковариационной матрицей (создается в 3_prepare_emb_matx_test.ipynb)

class RaspberryCamera:
    
    def __init__(self, resolution=(1920, 1080)):
        """
        Инициализация камеры с заданным разрешением.
        
        Args:
            resolution (tuple): Кортеж (ширина, высота) для установки разрешения камеры
        """
        self.camera = None  # Объект камеры Picamera2
        self.resolution = resolution  # Разрешение камеры
        self.is_initialized = False  # Флаг успешной инициализации
    
    def initialize(self):
        """
        Инициализация и настройка камеры.
        
        Returns:
            bool: True если инициализация прошла успешно, False в случае ошибки
        """
        try:
            self.camera = Picamera2()
            
            # Настройка конфигурации камеры
            config = self.camera.create_still_configuration(
                main={"size": self.resolution}  # Установка разрешения
            )
            self.camera.configure(config)
            
            # Запуск камеры
            self.camera.start()
            
            # Даем камере время на инициализацию
            time.sleep(2)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Ошибка при инициализации камеры: {str(e)}")
            return False
    
    def capture_image(self):
        """
        Получение одиночного кадра с камеры.
        
        Returns:
            numpy.ndarray: Изображение в формате numpy array или None в случае ошибки
        """
        if not self.is_initialized:
            print("Камера не инициализирована")
            return None
            
        try:
            return self.camera.capture_array()
        except Exception as e:
            print(f"Ошибка при получении изображения: {str(e)}")
            return None
    
    def save_image(self, image, directory="images"):
        """
        Сохранение полученного изображения в файл.
        
        Args:
            image: Изображение для сохранения
            directory (str): Директория для сохранения изображений
            
        Returns:
            str: Путь к сохраненному файлу или None в случае ошибки
        """
        try:
            # Создаем директорию, если она не существует
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Генерируем имя файла на основе текущего времени
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"{directory}/image_{timestamp}.jpg"
            filename = f"{directory}/image.jpg"
            
            # Сохраняем изображение
            self.camera.capture_file(filename)
            return filename
            
        except Exception as e:
            print(f"Ошибка при сохранении изображения: {str(e)}")
            return None
    
    def close(self):
        """
        Корректное закрытие камеры и освобождение ресурсов.
        """
        if self.is_initialized and self.camera:
            try:
                self.camera.close()
                self.is_initialized = False
            except Exception as e:
                print(f"Ошибка при закрытии камеры: {str(e)}")

    def __enter__(self):
        """
        Метод для поддержки контекстного менеджера (with statement).
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Метод для поддержки контекстного менеджера (with statement).
        """
        self.close()



def load_matrix(filename):
    """
    Загружает матрицу из файла
    
    Параметры:
    filename: str - имя файла для загрузки (с расширением .npy)
    
    Возвращает:
    numpy array - загруженная матрица
    """
    try:
        matrix = np.load(filename)  # Загрузка матрицы из файла
        print(f"Матрица успешно загружена из файла {filename}")
        return matrix
    except Exception as e:
        print(f"Ошибка при загрузке матрицы: {e}")
        return None
    
def load_embeddings_from_file(filename):
    try:
        with open(filename, 'rb') as file:
            embeddings = pickle.load(file)  # Загрузка словаря из файла
        print(f"Эмбеддинги успешно загружены из файла {filename}")
        return embeddings
    except Exception as e:
        print(f"Ошибка при загрузке эмбеддингов: {e}")
        return None

class Drone:
    def __init__(self, id, x, y, z):
        """
        Args:
            id: Идентификатор дрона (не используется в логике, но может быть полезен).
            x (float): Координата X дрона на карте.
            y (float): Координата Y дрона на карте.
            z (float): Координата Z (высота) дрона (не используется в текущей версии).
        """
        self.id = id
        self.x = x
        self.y = y
        self.z = z

class Encoder(nn.Module):

    def __init__(self, ENCODED_SPACE_DIM):
        """
        Args:
            ENCODED_SPACE_DIM (int): Размерность выходного эмбеддинга (латентного пространства).
        """
        super().__init__()
        
        self.encoded_space_dim = ENCODED_SPACE_DIM
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=2, padding=0),  # Входной слой: 3 канала (RGB), выход 8 каналов, ядро 4x4, шаг 2
            nn.ReLU(True),  # Функция активации ReLU
            nn.Conv2d(8, 16, 4, stride=2, padding=0),  # Скрытый слой: 8 -> 16 каналов, ядро 4x4, шаг 2
            nn.BatchNorm2d(16),  # Нормализация по батчу
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=0),  # Скрытый слой: 16 -> 32 канала, ядро 4x4, шаг 2
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),  # Скрытый слой: 32 -> 64 канала, ядро 4x4, шаг 2
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # Скрытый слой: 64 -> 128 каналов, ядро 4x4, шаг 2
            nn.ReLU(True)
        )
        
        # Слой для преобразования тензора в вектор (flattening)
        self.flatten = nn.Flatten(start_dim=1)
        # Полносвязная часть энкодера
        self.encoder_lin = nn.Sequential(
            nn.Linear(8 * 8 * 128, 1024),  # Вход: 8*8*128 (выход сверточной части для входа 300x300), выход: 1024
            nn.ReLU(True),
            nn.Linear(1024, ENCODED_SPACE_DIM)  # Выходной слой: 1024 -> ENCODED_SPACE_DIM (размерность эмбеддинга)
        )
        
    def forward(self, x):
        """
        Прямой проход через энкодер.
        
        Args:
            x (torch.Tensor): Входной тензор изображения (батч x каналы x высота x ширина).
        
        Returns:
            torch.Tensor: Выходной тензор эмбеддингов (батч x ENCODED_SPACE_DIM).
        """
        x = self.encoder_cnn(x)  # Прохождение через сверточные слои
        x = self.flatten(x)  # Преобразование в вектор
        x = self.encoder_lin(x)  # Прохождение через полносвязные слои
        return x 

def image_to_tensor(image):
    """
    Преобразует изображение из numpy array (BGR) в тензор PyTorch (RGB).
    
    Args:
        image (numpy.ndarray): Изображение в формате numpy array (BGR, от OpenCV или камеры).
    
    Returns:
        torch.Tensor: Тензор изображения (1 x C x H x W), готовый для подачи в нейронную сеть.
    """
    fpv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование цветового пространства из BGR (OpenCV) в RGB
    transform=transforms.ToTensor()  # Создание объекта преобразования для конвертации в тензор
    fpv_image = transform(fpv_image)  # Применение преобразования
    fpv_image = fpv_image.unsqueeze(0)  # Добавление размерности батча (unsqueeze(0))
    # print("Image shape:",fpv_image.shape)  # Вывод формы тензора (закомментировано для отладки)
    return fpv_image

def image_to_embeding(inputs):
     """
    Преобразует тензор изображения в эмбеддинг с помощью обученного энкодера.
    
    Args:
        inputs (torch.Tensor): Тензор изображения (1 x C x H x W).
    
    Returns:
        numpy.ndarray: Массив эмбеддингов (1 x ENCODED_SPACE_DIM) в виде numpy array.
    """
    encoder = Encoder(ENCODED_SPACE_DIM=256)  # Создание экземпляра энкодера с размерностью 256
    encoder.to(device)
    best_encoder = torch.load(ENCODER_MODEL_PATH, weights_only=True, map_location=device)  # Загрузка обученной модели энкодера
    encoder.load_state_dict(best_encoder)  # Загрузка весов в модель
    encoder.eval()  # Перевод модели в режим оценки (inference)
    with torch.no_grad():
        # Вычисление эмбеддинга без вычисления градиентов
        inputs = inputs.to(device)  # Перемещение входных данных на устройство
        coded= encoder(inputs)  # Получение эмбеддинга от энкодера
        coded = coded.cpu().numpy()
        print("Image coded emb shape:",coded.shape)
        return coded

def nearbyAreaCheck(drone, embedings, embeding_vector):
    """
    Проверяет, находится ли эмбеддинг в пределах заданного радиуса (NEARBY_DELTA) от текущей позиции дрона.
    
    Args:
        drone (Drone): Объект дрона с текущими координатами (x, y).
        embedings (dict): Словарь эмбеддингов (ключ: эмбеддинг, значение: [x, y]).
        embeding_vector (numpy.ndarray): Эмбеддинг для проверки.
    
    Returns:
        bool: True если эмбеддинг находится в пределах радиуса, иначе False.
    """
    # Получение координат, соответствующих эмбеддингу
    x = embedings[tuple(embeding_vector)][0]
    y = embedings[tuple(embeding_vector)][1]
    # Проверка условия близости по обеим координатам
    if abs(x-drone.x) < NEARBY_DELTA and abs(y-drone.y) < NEARBY_DELTA:
        return True
    else:
        return False


def process_image_to_square(image_array, target_size=TILE_SIZE):
    """
    Обрабатывает изображение: обрезает до квадрата и масштабирует до целевого размера.
    
    Args:
        image_array (numpy.ndarray): Исходное изображение в формате numpy array (1920x1080)
        target_size (int): Целевой размер изображения (по умолчанию 300x300)
    
    Returns:
        numpy.ndarray: Обработанное изображение размером target_size x target_size
    """
    try:
        # Преобразуем numpy array в объект PIL Image
        image = Image.fromarray(image_array)
        
        # Получаем текущие размеры
        width, height = image.size
        
        # Определяем размер для обрезки (берем минимальную сторону)
        crop_size = min(width, height)
        
        # Вычисляем координаты для центральной обрезки
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # Обрезаем изображение до квадрата
        square_image = image.crop((left, top, right, bottom))
        
        # Масштабируем до целевого размера
        resized_image = square_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Преобразуем обратно в numpy array
        result = np.array(resized_image)
        
        return result
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {str(e)}")
        return None



# Инициализация дрона с начальными координатами
drone = Drone(0, 333, 777, 0)

# Загрузка эмбеддингов и инверсной ковариационной матрицы
embedings = load_embeddings_from_file(EMBEDDINGS_PATH)  # Словарь эмбеддингов
inv_cov_matrix = load_matrix(MATRIX_PATH)  # Инверсная ковариационная матрица

while True:
    # get image from camera
    with RaspberryCamera() as camera:
        # Получаем изображение
        image = camera.capture_image()
        image = process_image_to_square(image)  # Обрабатываем изображение (обрезка до квадрата 300x300)
    
    fpv_image = image_to_tensor(image)  # Преобразуем изображение в тензор

    # image to encoder
    encoded_image = image_to_embeding(fpv_image).flatten()

    min_mahalanobis_distance = np.inf  # Минимальное расстояние Махаланобиса
    closest_embeding_vector = None  # Ближайший эмбеддинг
    # тут нужно потом сравнивать не со всеми а только с ближайшими
    for embeding_vector in  embedings.keys():
        if nearbyAreaCheck(drone, embedings, embeding_vector):
            mahalanobis_distance = distance.mahalanobis(encoded_image, embeding_vector, inv_cov_matrix)
            # print(mahalanobis_distance)
            if mahalanobis_distance < min_mahalanobis_distance:
                min_mahalanobis_distance = mahalanobis_distance
                closest_embeding_vector = embeding_vector
           
    # print("Min mahal: ",min_mahalanobis_distance)
    # print("New Drone coords: ",embedings[tuple(closest_embeding_vector)])
    # print("Real Drone coords: ", drone.x, drone.y)

    print(embedings[tuple(closest_embeding_vector)])
    
