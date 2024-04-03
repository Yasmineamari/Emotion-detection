import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

# Chemins relatifs aux données dans le répertoire du projet
train_path = 'data/train'
test_path = 'data/test'

# Configuration du générateur d'images pour l'entraînement
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),  # VGG16 prend des images de taille (224, 224)
    batch_size=32,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'  # VGG16 prend des images en couleur (RGB)
)

# Configuration du générateur d'images pour la validation
validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

# Création du modèle avec VGG16 pré-entraîné
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Ne pas entraîner les couches existantes du modèle pré-entraîné
for layer in base_model.layers:
    layer.trainable = False

# Création du modèle complet
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

# Compilation du modèle
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

# Répertoire du jeu de données d'entraînement
input_train_folder = 'data/train'


# Fonction pour égaliser l'histogramme en parallèle
def process_image(input_path):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)  # VGG16 prend des images en couleur (RGB)
    equalized_image = cv2.equalizeHist(image[:, :, 0])  # Prenez uniquement le canal rouge pour l'égalisation (une option parmi d'autres)
    equalized_image = cv2.merge([equalized_image, image[:, :, 1], image[:, :, 2]])  # Réassemblez les canaux pour obtenir une image couleur
    cv2.imwrite(input_path, equalized_image)


# Réduction du bruit en parallèle
override_noise_reduction(input_train_folder)

# Égalisation de l'histogramme en parallèle
override_histogram_equalization_parallel(input_train_folder)
