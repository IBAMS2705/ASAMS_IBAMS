import cv2
import numpy as np
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage


def process_images_to_firebase(uploaded_images,studentname):
    # Check if the app is not initialized before initializing it
    if not firebase_admin._apps:
        cred = credentials.Certificate("D:/PycharmProjects/IBAMS/Generated_servicekey.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': "ibams-944ab.appspot.com"
        })

    encodeImagesKnown = []
    studentid = []

    if os.path.exists("D:/PycharmProjects/IBAMS/EncodedFile.pkl"):
        with open("D:/PycharmProjects/IBAMS/EncodedFile.pkl", 'rb') as file:
            existing_data = pickle.load(file)
        encodeImagesKnown, studentid = existing_data

    def findencoding(imagesList):
        encodeList = []
        for image in imagesList:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                # Skip images without any detected faces
                continue
            encode = face_recognition.face_encodings(image, face_locations)[0]
            encodeList.append(encode)
        return encodeList

    # Create a Firebase Storage bucket reference
    bucket = storage.bucket()

    # Loop through uploaded images and process each one
    for index, image in enumerate(uploaded_images):
        # Read the image content from the UploadedFile
        image_content = np.asarray(bytearray(image.read()), dtype=np.uint8)

        # Decode the image content to a NumPy array
        image_np = cv2.imdecode(image_content, 1)

        studentid.append(studentname)

        encoded_image = findencoding([image_np])
        encodeImagesKnown.extend(encoded_image)
        print(f"Processed uploaded image: {index}")

    # Save the updated encodings and student IDs to a pickle
    encodeImagesKnown = [encodeImagesKnown, studentid]
    file = open("D:/PycharmProjects/IBAMS/EncodedFile.pkl", 'wb')
    pickle.dump(encodeImagesKnown, file)
    file.close()
    print("Encoding completed and file saved.")


