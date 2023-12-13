import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.base import BaseEstimator, RegressorMixin
from keras.optimizers import Adam

# Transformation des images pour les adapter au modèle
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((56, 56))
    img_array = np.array(img) / 255.0  # Normalisation des valeurs de pixel
    return img_array

# Wrapper class for Keras model
class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, optimizer='adam', dropout_rate=0.0, num_conv_layers=3, num_neurons=128, filter_size=(3, 3), learning_rate=0.001, pooling_rate=(2, 2)):
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.num_conv_layers = num_conv_layers
        self.num_neurons = num_neurons
        self.filter_size = filter_size
        self.learning_rate = learning_rate
        self.pooling_rate = pooling_rate
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()

        for _ in range(self.num_conv_layers):
            model.add(Conv2D(self.num_neurons, self.filter_size, input_shape=(56, 56, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=self.pooling_rate))
            model.add(Dropout(self.dropout_rate))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Dense(1, activation='linear'))

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse', 'RootMeanSquaredError'])

        return model

    def fit(self, X, y):
        # Preprocess X if needed
        images = np.array([preprocess_image(img_path) for img_path in X])

        # Fit the model
        self.model.fit(images, y)
        return self

    def predict(self, X):
        # Preprocess X if needed
        images = np.array([preprocess_image(img_path) for img_path in X])

        # Make predictions
        return self.model.predict(images)

# Charger les données et prétraiter
metadata = pd.read_csv("../data/train.csv")
popularity_score = metadata["Pawpularity"].values
images_ids = metadata["Id"].tolist()
image_paths = [f"../data/train/{id}.jpg" for id in images_ids]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(image_paths, popularity_score, test_size=0.2, random_state=42)

# Définir les hyperparamètres à optimiser
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.0, 0.2, 0.4],
    'num_conv_layers': [2, 3],
    'num_neurons': [64, 128, 256],
    'filter_size': [(3, 3), (5, 5)],
    'learning_rate': [0.001, 0.01, 0.1],
    'pooling_rate': [(2, 2), (3, 3)]
}

# Créer l'objet GridSearchCV avec le wrapper
grid = GridSearchCV(estimator=KerasRegressorWrapper(), param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3)

# Exécuter la recherche sur grille
grid_result = grid.fit(X_train, y_train)

# Afficher les résultats
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
