import numpy as np
import pandas as  pd
from PIL import Image
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from plot_functions import plot_data_train, plot_training_history
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from google.cloud import storage
import os
from keras.applications import VGG16
from sklearn.metrics import r2_score


# Transformation des images pour les adapter au modèle

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((56, 56))
    img_array = np.array(img) / 255.0
    return img_array



def build_transfer_learning_model(input_shape=(56, 56, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    predictions = Dense(1, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def get_metadata():
    metadata = pd.read_csv("data/csv/train.csv")
    return metadata[["Id", "Pawpularity"]]


def get_images(images_ids, images_path):
    images = []
    for id in images_ids:
        image_path = os.path.join(images_path, f"{id}.jpg")
        print(f"Checking existence of {image_path}")
        try:
            img_array = preprocess_image(image_path)
            images.append(img_array)
            print(f"Image {id} loaded successfully")
        except FileNotFoundError:
            print(f"Error: Image {id} not found.")
    return np.array(images)


def download_csv_from_gcs():
    # Téléchargez les fichiers CSV si le dossier de données n'existe pas
    bucket_name = "pet-finder"
    project_id = "pet-finder-407918"

    # Initialiser le client de stockage
    client = storage.Client(project=project_id)

    # Liste des fichiers CSV à télécharger
    csv_files_to_download = [
        'data/sample_submission.csv',
        'data/test.csv',
        'data/train.csv',
    ]

    # Créez le dossier local CSV s'il n'existe pas
    script_directory = os.getcwd()
    local_csv_folder = os.path.join(script_directory, 'data', 'csv')

    if not os.path.exists(local_csv_folder):
        os.makedirs(local_csv_folder)

    for csv_file_path in csv_files_to_download:
        blob = client.bucket(bucket_name).blob(csv_file_path)

        local_csv_destination = os.path.join(local_csv_folder, os.path.basename(csv_file_path))

        print(f"Checking existence of {local_csv_destination}")

        if not os.path.exists(local_csv_destination):
            blob.download_to_filename(local_csv_destination)
            print(f"CSV file {csv_file_path} downloaded to {local_csv_destination}")
        else:
            print(f"CSV file {csv_file_path} already exists in the local destination")

    print("CSV files downloaded successfully.")

def download_images_from_gcs():
    # Téléchargez les dossiers "train" et "test" si le dossier de données n'existe pas
    bucket_name = "pet-finder"
    project_id = "pet-finder-407918"

    # Initialiser le client de stockage
    client = storage.Client(project=project_id)

    # Liste des préfixes des fichiers à télécharger
    prefixes = ['data/train/', 'data/test/']

    download_needed = False  # Variable pour vérifier si le téléchargement est nécessaire

    for prefix in prefixes:
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        script_directory = os.getcwd()

        # Créez le dossier local de destination pour les images
        local_destination_folder = os.path.join(script_directory, prefix[:-1])

        if not os.path.exists(local_destination_folder):
            os.makedirs(local_destination_folder)
            download_needed = True

        # Si le téléchargement est nécessaire, procédez
        if download_needed:
            for blob in blobs:
                blob_name = blob.name[len(prefix):]
                local_destination = os.path.join(local_destination_folder, blob_name)

                print(f"Checking existence of {local_destination}")

                if not os.path.exists(local_destination):
                    blob.download_to_filename(local_destination)
                    print(f"File {blob_name} downloaded to {local_destination}")
                else:
                    print(f"File {blob_name} already exists in the local destination")
        else:
            print("Data already exists locally. No need to download.")

    print("Images downloaded successfully.")


def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lr


def train():
    if __name__ == "__main__":
        # Utilisation de la fonction pour télécharger les données dans le répertoire du script
        download_csv_from_gcs()
        # Utilisation de la fonction pour télécharger les images
        download_images_from_gcs()

    # Vérifier si le fichier 'data/train.csv' existe avant de le lire
    train_csv_path = 'data/csv/train.csv'
    if os.path.exists(train_csv_path):
        train_data = pd.read_csv(train_csv_path)
        plot_data_train(train_data)
        metadata = get_metadata()
        popularity_score = metadata["Pawpularity"].values
        print("Popularity scores acquired")
        images_ids = metadata["Id"].tolist()
        images = get_images(images_ids, 'data/train') #modifier si on veut utiliser test
        print("Images loaded")

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(images, popularity_score, test_size=0.2, random_state=42)

        transfer_model = build_transfer_learning_model()
        transfer_model.compile(optimizer=RMSprop(lr=0.0001), loss='mean_squared_error',
                               metrics=['mae', 'mse', 'RootMeanSquaredError'])
        #model.compile(optimizer=Adam(lr=0.0003), loss='mean_squared_error', metrics=['mae', 'mse', 'RootMeanSquaredError'])


        lr_scheduler = LearningRateScheduler(step_decay)
        early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=10, restore_best_weights=True)

        num_epochs = 10
        batch_size = 16

        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                     shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        datagen.fit(X_train)

        history = transfer_model.fit(datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
                            epochs=num_epochs, validation_data=(X_test, y_test),
                            callbacks=[early_stopping, lr_scheduler], workers=8)

        # Plot de l'historique d'entraînement
        plot_training_history(history)

        # Évaluer le modèle
        test_loss = transfer_model.evaluate(X_test, y_test)
        y_pred = transfer_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        print(f"Test loss: {test_loss}")
        print(f"R2 score for transfer learning model: {r2}")

        # Sauvegarder le modèle
        transfer_model.save("animal_adoption_model.h5")
        print("The model has been saved")
    else:
        print(f"Error: File '{train_csv_path}' not found.")

train()
