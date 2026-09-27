from scipy.spatial import distance
import numpy as np 
import pandas as pd 
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import pickle
from frame_transform import prepare_frame_for_model

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available')
else:
    print('CUDA is available!')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
TILE_STEP = 50
TILE_SIZE = 300
ENCODED_SPACE_DIM = 256
NEARBY_DELTA = 300
MAP_PATH = '../map/test_map_crop.png'
ENCODER_MODEL_PATH = '../models/best_encoder_02_08.pth'
EMBEDDINGS_PATH = "../embeddings.pkl"
MATRIX_PATH = "../inv_cov_matrix.npy"

def test_padding_stability(encoder, embeddings_db, tile_img, true_pos, ref_alt, P):
    """Критический тест: не смещает ли паддинг argmax поиска."""
    for h in [ref_alt * f for f in (0.6, 0.8, 1.0, 1.2, 1.5)]:
        frame = prepare_frame_for_model(tile_img, h, ref_alt, P)
        emb = encoder.predict(frame[None, ...])
        best = match_embedding(emb, embeddings_db)   # существующая логика поиска
        ok = (best.pos == true_pos)
        print(f"h={h:.1f}м  найдено={best.pos}  {'OK' if ok else 'СБОЙ'}")

def load_matrix(filename):
    """
    Загружает матрицу из файла
    
    Параметры:
    filename: str - имя файла для загрузки (с расширением .npy)
    
    Возвращает:
    numpy array - загруженная матрица
    """
    try:
        matrix = np.load(filename)
        print(f"Матрица успешно загружена из файла {filename}")
        return matrix
    except Exception as e:
        print(f"Ошибка при загрузке матрицы: {e}")
        return None
    
def load_embeddings_from_file(filename):
    try:
        with open(filename, 'rb') as file:
            embeddings = pickle.load(file)
        print(f"Эмбеддинги успешно загружены из файла {filename}")
        return embeddings
    except Exception as e:
        print(f"Ошибка при загрузке эмбеддингов: {e}")
        return None

class Drone:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z

class Encoder(nn.Module):

    def __init__(self, ENCODED_SPACE_DIM):
        super().__init__()
        
        self.encoded_space_dim = ENCODED_SPACE_DIM
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 4, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(8 * 8 * 128, 1024),
            nn.ReLU(True),
            nn.Linear(1024, ENCODED_SPACE_DIM)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        #z_mean = self.encoder_lin(x)
        #z_log_var = self.encoder_lin(x)
        #N = torch.normal(0, 1, size=(z_log_var.size()[0], self.encoded_space_dim))
        #N = N.to(device)
        #print('N: ', N.shape, ' z_log_var: ', z_log_var.shape)
        return x #(torch.exp(z_log_var / 2) * N + z_mean), z_mean, z_log_var

def image_to_tensor(image):
    # ----------------------------
    # пока использую статичное изображение, потом буду получать от камеры
    x = drone.x
    y = drone.y
    full_map = cv2.imread(MAP_PATH)
    fpv_image = full_map[y:y+TILE_SIZE, x:x+TILE_SIZE]
    # ----------------------------
    fpv_image = cv2.cvtColor(fpv_image, cv2.COLOR_BGR2RGB)
    transform=transforms.ToTensor()
    fpv_image = transform(fpv_image)
    fpv_image = fpv_image.unsqueeze(0)
    # print("Image shape:",fpv_image.shape)
    return fpv_image

def image_to_embeding(inputs):
     
    encoder = Encoder(ENCODED_SPACE_DIM=256)
    encoder.to(device)
    best_encoder = torch.load(ENCODER_MODEL_PATH, weights_only=True)
    encoder.load_state_dict(best_encoder)  
    encoder.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        coded= encoder(inputs)
        coded = coded.cpu().numpy()
        print("Image coded emb shape:",coded.shape)
        return coded

def nearbyAreaCheck(drone, embedings, embeding_vector):
    x = embedings[tuple(embeding_vector)][0]
    y = embedings[tuple(embeding_vector)][1]
    if abs(x-drone.x) < NEARBY_DELTA and abs(y-drone.y) < NEARBY_DELTA:
        return True
    else:
        return False



# drone start coords
drone = Drone(0, 333, 777, 0)

embedings = load_embeddings_from_file(EMBEDDINGS_PATH) 
inv_cov_matrix = load_matrix(MATRIX_PATH)

# while True:
for i in range(1):
    # get image from camera
    fpv_image = image_to_tensor(drone)

    # image to encoder
    encoded_image = image_to_embeding(fpv_image).flatten()

    min_mahalanobis_distance = np.inf
    closest_embeding_vector = None
    # тут нужно потом сравнивать не со всеми а только с ближайшими
    for embeding_vector in  embedings.keys():
        if nearbyAreaCheck(drone, embedings, embeding_vector):
            mahalanobis_distance = distance.mahalanobis(encoded_image, embeding_vector, inv_cov_matrix)
            # print(mahalanobis_distance)
            if mahalanobis_distance < min_mahalanobis_distance:
                min_mahalanobis_distance = mahalanobis_distance
                closest_embeding_vector = embeding_vector

    print("Min mahal: ",min_mahalanobis_distance)
    print("New Drone coords: ",embedings[tuple(closest_embeding_vector)])
    print("Real Drone coords: ", drone.x, drone.y)
