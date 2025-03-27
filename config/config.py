import os

BASE_PATH = "dataset"

TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT, "class.h5")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot")

IMG_SIZE = (150,150)

INIT_LR = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 32

CLASS_NAMES = ["amarelo", "verde", "vermelho"]  