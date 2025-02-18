import os

import cv2
import face_recognition
import pandas as pd


def loadDataFromFolder(folder, metadataFile):
    images = []
    labels = []
    metadata = []

    metadataDataframe = pd.read_excel(metadataFile)
    metadataDictionary = metadataDataframe.set_index("Unique Identifiers").T.to_dict("list")

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = face_recognition.load_image_file(os.path.join(folder, filename))
            encoding = face_recognition.face_encodings(img)[0]
            images.append(encoding)
            labels.append(filename.split(".")[0])
            identifier = filename.split(".")[0]
            if identifier in metadataDictionary:
                metadata.append(metadataDictionary[identifier])
            else:
                metadata.append(["Unknown", "Unknown", "Unknown"])
    
    return images, labels, metadata

trainingDataFolder = "D:\\workspace\\facialytics\\training-data";
metadataFile = "D:\\workspace\\facialytics\\training-data\\data.xlsx";

knownFaceEncoding, knownFaceNames, metadata = loadDataFromFolder(trainingDataFolder, metadataFile)