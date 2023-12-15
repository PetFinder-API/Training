import numpy as np
import pandas as pd
from PIL import Image
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from Python_file.plot_functions import plot_data_train, plot_training_history
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop


# Transformation des images pour les adapter au modèle
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((56, 56))
    img_array = np.array(img) / 255.0  # Normalisation des valeurs de pixel
    return img_array


def build_model():
    # To improve the model perforxmance, you could look into transfer learning
    # to use existing model and only train the n-last layers

    model = Sequential()

    model.add(Conv2D(32, (3, 3),input_shape=(56, 56, 3), activation='leaky_relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))


    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))


    model.add(Conv2D(128, (3, 3), activation='leaky_relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='leaky_relu'))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='leaky_relu'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='linear'))

    return model


def get_metadata():
    metadata = pd.read_csv("../data/train.csv")
    return metadata[["Id", "Pawpularity"]]


def get_images(ids):
    images_path = "../data/train/"
    return np.array([preprocess_image(images_path + "/" + id + ".jpg") for id in ids])


def download_images_from_gcs():
    # download if data folder does not exist
    pass

def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lr


def train():
    download_images_from_gcs()
    train_data = pd.read_csv('../data/train.csv')
    plot_data_train(train_data)
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

    #model.compile(optimizer=Adam(lr=0.0003), loss='mean_squared_error', metrics=['mae', 'mse', 'RootMeanSquaredError'])
    model.compile(optimizer=RMSprop(lr=0.0001), loss='mean_squared_error', metrics=['mae', 'mse', 'RootMeanSquaredError'])

    lr_scheduler = LearningRateScheduler(step_decay)
    early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=10, restore_best_weights=True)

    num_epochs = 10
    batch_size = 32

    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)


    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
                        epochs=num_epochs, validation_data=(X_test, y_test),
                        callbacks=[early_stopping, lr_scheduler], workers=4)

    # Plot de l'historique d'entraînement
    plot_training_history(history)

    # Évaluer le modèle
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}")

    # Sauvegarder le modèle
    model.save("animal_adoption_model.h5")
    print("The model has been saved")


train()
