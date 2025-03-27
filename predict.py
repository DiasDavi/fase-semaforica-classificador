import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from config import config
import argparse
import os

# Argumentos para receber a imagem de entrada
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem de entrada")
args = vars(ap.parse_args())

image_path = args["image"]

# Verifica se a imagem existe
if not os.path.exists(image_path):
    print(f"[ERRO] A imagem '{image_path}' não foi encontrada.")
    exit(1)

print("[INFO] Carregando modelo...")
model = keras.models.load_model(config.MODEL_PATH)
print("[SUCESSO] Modelo carregado com sucesso.\n")

def preprocess_image(image_path):
    """Carrega e pré-processa a imagem para o modelo."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMG_SIZE)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    class_index = np.argmax(predictions)  
    confidence = np.max(predictions)  
    class_name = config.CLASS_NAMES[class_index]

    print(f"[RESULTADO] Classe prevista: {class_name} ({confidence:.2f})")
    return class_name, confidence

predict(image_path)
