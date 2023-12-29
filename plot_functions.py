import matplotlib.pyplot as plt
import seaborn as sns

def plot_data_train(train_data):
    plt.figure(figsize=(8, 6))

    sns.histplot(train_data['Pawpularity'],
                 bins=25,
                 kde=True
                 )

    plt.title('Distribution of Pawpularity')
    plt.show()

    plt.figure(figsize=(8, 6))

    sns.boxplot(x=train_data['Pawpularity'])

    plt.title('Box Plot of Pawpularity Distribution')
    plt.xlabel('Pawpularity')
    plt.show()

def plot_training_history(history):
    loss = history.history['loss']
    mae = history.history['mae']
    mse = history.history['mse']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 4))

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
