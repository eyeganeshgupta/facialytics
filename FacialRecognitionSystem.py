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

videoCapture = cv2.VideoCapture(0)
while True:
    ret, frame = videoCapture.read()
    faceLocations = face_recognition.face_locations(frame)
    faceEncodings = face_recognition.face_encodings(frame, faceLocations)

    for faceEncoding in faceEncodings:
        matches = face_recognition.compare_faces(knownFaceEncoding, faceEncoding)
        name = "Unknown"
        fullName = "Unknown"
        age = "Unknown"
        gender = "Unknown"

        if True in matches:
            firstMatchIndex = matches.index(True)
            name = knownFaceNames[firstMatchIndex]
            fullName, age, gender = metadata[firstMatchIndex]

            cv2.putText(frame, "Identity verified, access granted", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 2555, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Identity not verified, access denied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 2555, 0), 2, cv2.LINE_AA)
        
        top, right, bottom, left = face_recognition.face_locations(frame)[0]
        cv2.rectangle(frame, (left, top), (right, bottom + 20), (0, 0, 255), 2)

        cv2.putText(frame, f"Name: {fullName}", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, f"Age: {age}", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, f"Gender: {gender}", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
