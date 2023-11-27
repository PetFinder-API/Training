import pymongo
import pandas as pd
from fastai.vision.all import *
import cv2
from PIL import Image
import os

client = pymongo.MongoClient("mongodb://localhost:27017/")

db = client["CSVPetFinder"]

train_data = db["train_data"]

dataset_path = Path("C:/Users/conta/PycharmProjects/Projet_ESME3/")

picture_origin = "C:/Users/conta/PycharmProjects/Projet_ESME3/train"

dimensioned_picture = "C:/Users/conta/PycharmProjects/Projet_ESME3/train_picture_upgrade"

if not os.path.exists(dimensioned_picture):
    os.mkdir(dimensioned_picture)

image_extensions = [".jpg", ".png", ".jpeg"]

for document in train_data.find():
    id = document["Id"]
    path = str(dataset_path / "train_data" / (id + ".jpg"))
    train_data.update_one({"Id": id}, {"$set": {"path": path}})
    for filename in os.listdir(picture_origin):
        if any(filename.endswith(extension) for extension in image_extensions):
            if id in filename:
                img = Image.open(os.path.join(picture_origin, filename))

                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img_normalized = cv2.normalize(img_cv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)

                img_normalized = cv2.resize(img_normalized, (224, 224))

                # Modifiez le chemin de sauvegarde pour "train_picture_upgrade"
                img_normalized_pil = Image.fromarray((img_normalized * 255).astype('uint8'))
                img_normalized_pil.save(os.path.join(dimensioned_picture, filename))

print("Prétraitement des images terminé.")

