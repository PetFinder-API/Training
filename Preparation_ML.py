from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split


# Chemin vers le dossier contenant les images prétraitées
dimensioned_picture = "C:/Users/conta/PycharmProjects/Projet_ESME3/train_picture_upgrade"

# Charger les données et appliquer la transformation
image_paths = [os.path.join(dimensioned_picture, filename) for filename in os.listdir(dimensioned_picture) if filename.endswith(".jpg")]

# Transformation des images pour les adapter au modèle
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((56, 56))
    img_array = np.array(img) / 255.0  # Normalisation des valeurs de pixel
    return img_array


X = [preprocess_image(img_path) for img_path in image_paths]

#Appeler Mongo pour mettre la colonne "adoptabilité"
y = [1] * len(X)  # Ajuster la gestion de la cible en fonction de votre ensemble de données

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir les listes en tableaux NumPy
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Définir le modèle
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(56, 56, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
num_epochs = 10
model.fit(X_train, y_train, epochs=num_epochs, batch_size=64, shuffle=True)

# Évaluer le modèle
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Sauvegarder le modèle
model.save("animal_adoption_model.h5")
