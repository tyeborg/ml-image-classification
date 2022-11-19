# Download the required packages.
# pip install mtcnn
# pip install tensorflow

import numpy as np
import pandas as pd
import cv2 as cv
from mtcnn.mtcnn import MTCNN

# Create a class that extracts a discernable face from image.
class FaceDetector:
    def __init__(self, img):
        self.detector = MTCNN()
        self.img = img
        # Locate the face within the input frame.
        self.face = self.detector.detect_faces(self.img)
        self.num_faces = self.receive_number_of_faces()
    
    # Create a method that returns the number of faces in input image.
    def receive_number_of_faces(self):
        return len(self.face)

    # Create a method that returns a given face from the image.
    def extract_face(self, img, face):
        # Draw a box around the discovered face.
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height 

        cv.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

        # Structure the image to the desired format.
        face_image = img[y:y2, x:x2]
        cv.imwrite('../results/detected.jpg', img)

        return face_image

    # Create a method that rescales the image of the extracted face.
    def scale_face_image(self, face_image):
        # Reduce the depth of the image to 1.
        face_image2 = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)

        # Resize the image to shape (32, 32).
        face_image3 = cv.resize(face_image2, (32, 32))

        # Normalize the image.
        face_image4 = face_image3/255

        # Adding image to 4D.
        face_image5 = np.expand_dims(face_image4, 0)

        # Return the frame for a Deepfake Analysis.
        return face_image5