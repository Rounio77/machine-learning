import os
import cv2
import face_recognition


# Initialize lists to store known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []


# Function to load known faces and their encodings from a specified directory
def load_known_faces(directory):
    for filename in os.listdir(directory):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            # Load the image
            image = face_recognition.load_image_file(image_path)
            # Get the face encodings for the image
            encodings = face_recognition.face_encodings(image)
            if encodings:
                # Append the encoding and the name (without extension) to the lists
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])


# Directory containing images of known individuals
known_faces_directory = "known_faces"
# Load known faces from the directory
load_known_faces(known_faces_directory)


# Function to detect and recognize faces in an input image
def recognize_faces(image_path):
    if not os.path.isfile(image_path):
        print(f"File '{image_path}' does not exist.")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}.")
        return

    # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face locations in the image
    face_locations = face_recognition.face_locations(rgb_image)
    # Get face encodings for the detected faces
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Loop through each detected face and its encoding
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_ITALIC
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the image with detected faces and names
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'known_faces/angelina jolie.png'
recognize_faces(image_path)
