import os
from concurrent.futures import ThreadPoolExecutor

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

bucket_name = "pet-finder"
project_id = "pet-finder-407918"
script_directory = os.getcwd()

# Initialiser le client de stockage
client = storage.Client(project=project_id)


def download_csv_from_gcs():
    """
    Téléchargez les fichiers CSV si le dossier de données n'existe pas
    """

    # Liste des fichiers CSV à télécharger
    csv_files_to_download = [
        'data/sample_submission.csv',
        'data/test.csv',
        'data/train.csv',
    ]

    # Créez le dossier local CSV s'il n'existe pas
    local_csv_folder = os.path.join(script_directory, 'data', 'csv')

    if not os.path.exists(local_csv_folder):
        os.makedirs(local_csv_folder)

    for csv_file_path in csv_files_to_download:
        blob = client.bucket(bucket_name).blob(csv_file_path)

        local_csv_destination = os.path.join(local_csv_folder, os.path.basename(csv_file_path))

        if not os.path.exists(local_csv_destination):
            blob.download_to_filename(local_csv_destination)
            print(f"CSV file {csv_file_path} downloaded to {local_csv_destination}")
        else:
            print(f"CSV file {csv_file_path} already exists in the local destination")

    print("CSV files downloaded successfully.")


def download_images_from_gcs():
    """
    Téléchargez les dossiers "train" et "test" si le dossier de données n'existe pas
    """

    # Initialiser le client de stockage
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    # Liste des préfixes des fichiers à télécharger
    prefixes = ['data/train/', 'data/test/']

    for prefix in prefixes:
        blobs = bucket.list_blobs(prefix=prefix)

        # Créez le dossier local de destination pour les images
        local_destination_folder = os.path.join(script_directory, prefix)

        # if the images download hasn't happened yet
        if not os.path.exists(local_destination_folder):
            os.makedirs(local_destination_folder)

            with ThreadPoolExecutor() as executor:
                for blob in blobs:
                    executor.submit(download_single_image, blob, local_destination_folder, prefix)
        else:
            print("Data already exists locally. No need to download.")

    print("Images downloaded successfully.")


def download_single_image(blob, local_destination_folder, prefix):
    blob_name = blob.name[len(prefix):]
    local_destination = os.path.join(local_destination_folder, blob_name)
    if not os.path.exists(local_destination):
        blob.download_to_filename(local_destination)
        print(f"File {blob_name} downloaded to {local_destination}")
    else:
        print(f"File {blob_name} already exists in the local destination")


download_images_from_gcs()
