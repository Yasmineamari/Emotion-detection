import os
import Augmentor

# Define the paths and labels

labels = sorted(os.listdir(train_path))

# Find the maximum number of samples among all classes
max_samples = max(len(os.listdir(os.path.join(train_path, label))) for label in labels)

# Augment each class to match the maximum number of samples
for label in labels:
    p = Augmentor.Pipeline(os.path.join(train_path, label))

    # Calculate the number of additional samples needed
    num_samples_to_add = max_samples - len(p.augmentor_images)

    # If there are fewer samples than the maximum, augment the class
    if num_samples_to_add > 0:
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)

        # Sample the pipeline to generate augmented images
        p.sample(num_samples_to_add)


import os
import pathlib


train_path = "D:/renconaissance facial/train"  l

# Liste des classes
labels = sorted(os.listdir(train_path))

for i in labels:
    # Chemin du dossier d'entrée
    directory = pathlib.Path(train_path) / i
    
    # Chemin du dossier de sortie
    directory_out = pathlib.Path(train_path) / i / 'output'
    
    # Utilisez os.path.join pour construire les chemins complets
    files_in = [file for file in os.listdir(os.path.join(directory)) if os.path.isfile(os.path.join(directory, file))]
    files_out = [file for file in os.listdir(os.path.join(directory_out)) if os.path.isfile(os.path.join(directory_out, file))]
    
    
    
    # Obtenez les longueurs de chaque liste
    length = len(files_in)
    length_out = len(files_out)
    
    # Calculez le total
    length_tot = length + length_out
    
    print(f'{i} has {length_tot} samples (input: {length}, output: {length_out}).')

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
train_path = "D:/renconaissance facial/train"
test_path = "D:/renconaissance facial/test"
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