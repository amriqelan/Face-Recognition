# Face Recognition
The Face Recognition App is a Python-based application that utilizes the face_recognition library and OpenCV to perform real-time face recognition on a video stream. The app allows you to capture faces, associate them with names, and recognize those faces in subsequent frames of the video.

# Notes
The captured faces can now be recognized even when the application is run again. The code has been enhanced to store the face encodings and associated names in a file or a database for future use.

To save the face encoding and associated name, you can consider using a file or a database. For example, you can create a CSV file named `face_data.csv` in the `face_encodings` directory. Each row in the CSV file can represent a face, with the columns containing the face encoding and the associated name. When the application starts, load the face encodings and associated names from the CSV file and store them in memory. During face recognition, compare the captured face encoding with the stored encodings to determine the recognized faces.

To implement this functionality:

1. Create a folder named `face_encodings` in the same directory as the script.
2. Choose a storage format such as a CSV file, JSON, or a database.
3. Store the face encodings and associated names in the chosen format, saving them to the `face_encodings` folder.
4. When the application starts, load the face encodings and associated names from the storage file or database and store them in memory.
5. During face recognition, compare the captured face encoding with the stored encodings to determine the recognized faces.

Ensure that the `images` and `face_encodings` folders exist before running the application. The code provided uses a simple CSV file format to store the face data, but you can explore other storage options such as JSON or a database based on your specific requirements.
