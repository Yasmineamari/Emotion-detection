import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale'
)

# Configuration du générateur d'images pour la validation
validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale'
)

# Création du modèle CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

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
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(image)
    cv2.imwrite(input_path, equalized_image)

# Fonction pour réduire le bruit en parallèle
def override_noise_reduction(input_folder, kernel_size=(5, 5)):
    for emotion_folder in os.listdir(input_folder):
        emotion_path = os.path.join(input_folder, emotion_folder)

        for filename in os.listdir(emotion_path):
            if filename.endswith('.jpg'):
                input_path = os.path.join(emotion_path, filename)

                # Appliquer un flou gaussien pour réduire le bruit
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

                # Sauvegarder l'image floutée, en remplaçant l'original
                cv2.imwrite(input_path, blurred_image)

# Prétraitement des images - Égalisation de l'histogramme
def override_histogram_equalization_parallel(input_folder, num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for emotion_folder in os.listdir(input_folder):
            emotion_path = os.path.join(input_folder, emotion_folder)

            image_files = [filename for filename in os.listdir(emotion_path) if filename.endswith('.jpg')]

            for filename in image_files:
                input_path = os.path.join(emotion_path, filename)

                # Soumettre la tâche de traitement d'image à l'exécuteur
                futures.append(executor.submit(process_image, input_path))

        # Attendre la fin de toutes les tâches
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Erreur lors du traitement de l'image : {e}")

# Réduction du bruit en parallèle
override_noise_reduction(input_train_folder)

