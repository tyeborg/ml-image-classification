import os
import numpy as np 
import pandas as pd
import cv2 as cv
from keras.models import load_model
import tensorflow as tf

# Import classes.
from face_detector import FaceDetector

def retrieve_user_image():
    base_path = '../test/'
    # Declare boolean variable for validation purposes.
    invalid = True

    while invalid == True:
        # Ask the user for the image file.
        input_img = input("\nPlease enter image file ('q' to exit): ")

        # Determine if the user wants to exit early.
        if input_img == 'q' or input_img == 'Q':
            return input_img
        
        # Create the img path.
        input_img = base_path + input_img

        if os.path.exists(input_img) == False:
            # Notify the user of the file not existing within your current path.
            print(f'[*] \'{input_img}\' does not exist.')
            # Loop until a valid coordinate is inputted.
            invalid = True
        else:
            # Read the image.
            input_img = cv.imread(input_img)
            # Exit the while loop.
            break
    
    return input_img

# Create a method that'll ask the user if they want to classify another image.            
def initiate_redo_prompt():
    # Ask the user if they would like to enter another image.
    redo = input("\nWould you like to classify another image? (Yes or No): ")
    # Convert the letters within input to all lowercase.
    redo = redo.lower()

    return redo

# Create a method that'll give a smiling prediction based off image.
def receive_prediction(model, img):
    # Reduce the dimension of the result.
    prediction = model.predict(img).flatten()
    # Round the prediction to the nearest integer.
    prediction = round(prediction[0])

    return prediction

def classify_individual_face(model, detector, user_img, face):
    # Extract the face from the image.
    img = detector.extract_face(user_img, face)
    # Scale the image to the detected face.
    scaled_img = detector.scale_face_image(img)

    # Classify the following face as either SMILING or NOT SMILING.
    result = receive_prediction(model, scaled_img)

    return result, img

def classify_multiple_faces(model, detector, user_img):
    # Initialize a smile/no smile counter to keep track of their respective amounts.
    face_counter, smile_counter, no_smile_counter = 0, 0, 0

    for face in detector.face:
        face_counter += 1
        result = classify_individual_face(model, detector, user_img, face)

        if result[0] == 0:
            no_smile_counter += 1
            cv.imwrite('../results/not_smiling/face{}.jpg'.format(face_counter), result[1])
        else:
            smile_counter += 1
            cv.imwrite('../results/smiling/face{}.jpg'.format(face_counter), result[1])

    print(f'[*] There are {smile_counter} people smiling')
    print(f'[*] There are {no_smile_counter} people NOT smiling')

def remove_files(mydir):
    for f in os.listdir(mydir):
        os.remove(os.path.join(mydir, f))

def main():
    # Load the saved model.
    model = tf.keras.models.load_model('../model/smile-model.h5')

    # Declare boolean variables for looping/validation purposes.
    loop1 = True
    loop2 = True

    while loop1 == True:
        # Clear the smiling and not_smiling directories in 'results' folder.
        smile_path = '../results/smiling/'
        no_smile_path = '../results/not_smiling/'
        remove_files(smile_path)
        remove_files(no_smile_path)

        # Receive the input image from the user.
        user_img = retrieve_user_image()
        
        # Exit the program if 'q' or 'Q' was entered.
        if user_img == 'q' or user_img == 'Q':
            break

        # Detect faces within the user input image.
        face_detector = FaceDetector(user_img)

        # Provide action if the detector detects only one face...
        if face_detector.num_faces == 1:
            print(f'[*] Detected 1 face in image')
            # Call upon the classify_individual_face() function to handle one face.
            result = classify_individual_face(model, face_detector, user_img, face_detector.face[0])

            if result[0] == 0:
                print("[*] The individual in this image is NOT smiling")
                cv.imwrite('../results/not_smiling/face.jpg', result[1])
            else:
                print("[*] The individual in this image is smiling")
                cv.imwrite('../results/smiling/face.jpg', result[1])
        # Provide action if the detector detects more than one face.
        elif face_detector.num_faces > 1:
            print(f'[*] Detected {face_detector.num_faces} faces in image')
            # Call upon the classify_multiple_faces() function to handle multiple faces.
            classify_multiple_faces(model, face_detector, user_img)
        else:
            # Notify the user that a face could not be found in the image.
            print("[*] Could not find a discernable face in the image.")
            continue

        # Create another while loop to validate an upcoming prompt.
        while loop2 == True:
            redo = initiate_redo_prompt()

            # Determine if the user entered 'Yes' or 'No'.
            if redo == "yes":
                # If 'Yes' was entered, return to the longitude and latitude prompts.
                loop1 = True
                # Exit the nested while loop.
                break
            # If 'No' was entered by the user...
            elif redo == "no":
                # Ensure to not return to either of the prompts.
                loop1 = False
                loop2 = False
                # Exit the main loop and nested loop.
                break
            else:
                # Let the user know that only 'Yes' or 'No' responses are acceptable.
                print("[*]'Yes' or 'No' responses ONLY")
                # Return to the redo prompt.
                loop2 = True

    # Exit the program...
    print("Exiting the program...")

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\nSomething went wrong...")