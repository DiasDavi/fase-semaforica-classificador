from config import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print("[INFO] Carregando imagens de treino...")
train_ds = keras.utils.image_dataset_from_directory(
    config.TRAIN_PATH,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode="categorical" 
)
print("[SUCESSO] Carregado com sucesso.\n")

print("[INFO] Carregando imagens de teste...")
test_ds = keras.utils.image_dataset_from_directory(
    config.TEST_PATH,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    label_mode="categorical"
)
print("[SUCESSO] Carregado com sucesso.\n")

print("[INFO] Normalizando imagens...")
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
print("[SUCESSO] Imagens normalizadas com sucesso.\n")

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("[INFO] Treinando o modelo...")
history = model.fit(train_ds, validation_data=test_ds, epochs=config.NUM_EPOCHS)
test_loss, test_acc = model.evaluate(test_ds)


print("[INFO] Salvando o modelo...")
model.save(config.MODEL_PATH, save_format="h5")
print("[SUCESSO] Modelo salvo com sucesso.\n")

print("[INFO] Gerando gráficos do treinamento...")
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.savefig(config.PLOT_PATH)
plt.show()
print("[SUCESSO] Gráficos salvos com sucesso.\n")