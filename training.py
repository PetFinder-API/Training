import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,BatchNormalization, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
from keras.metrics import RootMeanSquaredError


# Transformation des images pour les adapter au modèle
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((56, 56))
    img_array = np.array(img) / 255.0  # Normalisation des valeurs de pixel
    return img_array

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip= True,
    fill_mode='nearest'
)
# Ajouter TensorBoard comme callback
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)


def build_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(56, 56, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='linear'))

    return model


def get_metadata():
    metadata = pd.read_json("train_data.json")
    return metadata[["Id", "Pawpularity"]].head(1000)


def get_images(ids):
    images_path = "train/"
    return np.array([preprocess_image(images_path + "/" + id + ".jpg") for id in ids])


def plot_training_history(history):
    loss = history.history['loss']
    mae = history.history['mae']
    mse = history.history['mse']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 4))

    # Plot de la perte
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot de la MAE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, mae, label='Training MAE')
    plt.title('Training MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # Plot de la MSE
    plt.subplot(1, 3, 3)
    plt.plot(epochs, mse, label='Training MSE')
    plt.title('Training MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 20:
        lr *= 0.5
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)



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
    # Utiliser 'mean_squared_error' pour la régression
    # Compiler le modèle avec le taux d'apprentissage réduit
    model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error', metrics=['mae', 'mse', RootMeanSquaredError()])


    early_stopping = EarlyStopping(monitor='val_root_mean_squared_error',patience=10,restore_best_weights=True)

    # Entraîner le modèle
    num_epochs = 10


    #Essayer batch_size 64 après (ne pas oublié tensorboard)
    history=model.fit(X_train,y_train,epochs=num_epochs,batch_size=32,validation_data=(X_test, y_test),callbacks=[early_stopping])

    # Plot de l'historique d'entraînement
    plot_training_history(history)

    # Évaluer le modèle
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}")

    # Sauvegarder le modèle
    model.save("animal_adoption_model.h5")
    print("The model has been saved")


train()
