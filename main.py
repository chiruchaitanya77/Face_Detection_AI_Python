# import cv2
# import numpy as np
# import face_recognition
# import os
#
# path = 'training'
# images = []
# classNames = []
# folderNames = []
#
# # Traverse the directory tree to load images and their folder names
# for root, dirs, files in os.walk(path):
#     for file in files:
#         if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
#             file_path = os.path.join(root, file)
#             currentImg = cv2.imread(file_path)
#             images.append(currentImg)
#             # Append the folder name instead of file name
#             folderName = os.path.basename(root)  # Extract folder name
#             classNames.append(folderName)  # Save folder name
#             folderNames.append(folderName)
#
# print(f"Classes: {classNames}")
#
# def findEncodings(images):
#     encodingList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodings = face_recognition.face_encodings(img)  # Get all encodings for the image
#         if len(encodings) > 0:  # Check if at least one encoding is found
#             encodingList.append(encodings[0])  # Use the first encoding
#         else:
#             print("Warning: No face found in one of the images. Skipping it.")
#     return encodingList
#
# # Encode all known images
# encodeListKnown = findEncodings(images)
# print('Encoding Complete')
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     success, img = cap.read()
#     imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
#
#     # Detect faces in the current frame
#     facesCurrFrame = face_recognition.face_locations(imgs)
#     encodesCurrFrame = face_recognition.face_encodings(imgs, facesCurrFrame)
#
#     for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#         print(faceDis)
#         matchIndex = np.argmin(faceDis)
#
#         if matches[matchIndex]:
#             # Get the folder name of the matched encoding
#             name = folderNames[matchIndex].upper()
#             print(name)
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#
#     cv2.imshow('Webcam', img)
#     cv2.waitKey(1)

import cv2
import numpy as np
import face_recognition
import os
import pickle

# Path configurations
training_path = 'training'
output_path = 'output/encodings.pkl'

# Function to find encodings
def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodingList.append(encodings[0])  # Use the first encoding
        else:
            print("Warning: No face found in one of the images. Skipping it.")
    return encodingList

# Load encodings from pickle if available
if os.path.exists(output_path):
    print("Loading encodings from pickle file...")
    with open(output_path, 'rb') as file:
        encodeListKnown, folderNames = pickle.load(file)
else:
    print("Generating encodings...")
    images = []
    folderNames = []

    # Traverse the directory tree to load images and their folder names
    for root, dirs, files in os.walk(training_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
                file_path = os.path.join(root, file)
                currentImg = cv2.imread(file_path)
                images.append(currentImg)
                folderName = os.path.basename(root)  # Extract folder name
                folderNames.append(folderName)

    print(f"Classes: {set(folderNames)}")  # Print unique folder names

    # Compute encodings
    encodeListKnown = findEncodings(images)

    # Save encodings and folder names to pickle
    print("Saving encodings to pickle file...")
    with open(output_path, 'wb') as file:
        pickle.dump((encodeListKnown, folderNames), file)

cap = cv2.VideoCapture(0)

# Perform face detection
while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    facesCurrFrame = face_recognition.face_locations(imgs)
    encodesCurrFrame = face_recognition.face_encodings(imgs, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            # Get the folder name of the matched encoding
            name = folderNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
