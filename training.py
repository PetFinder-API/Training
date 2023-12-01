import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split


# Transformation des images pour les adapter au modèle
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((56, 56))
    img_array = np.array(img) / 255.0  # Normalisation des valeurs de pixel
    return img_array


def build_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(56, 56, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_metadata():
    metadata = pd.read_csv("data/train.csv")
    return metadata[["Id", "Pawpularity"]].head(1000)


def get_images(ids):
    images_path = "data/train"
    return np.array([preprocess_image(images_path + "/" + id + ".jpg") for id in ids])


def train():
    # Appeler Mongo pour mettre la colonne "adoptabilité"
    metadata = get_metadata()
    popularity_score = metadata["Pawpularity"].values
    print("Popularity scores acquired")
    images_ids = metadata["Id"].tolist()
    images = get_images(images_ids)
    print("Images loaded")

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(images, popularity_score, test_size=0.2, random_state=42)

    # Définir le modèle
    model = build_model()

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
    print("The model has been saved")


train()
