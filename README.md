# Face-Recognition

Face Recognition App
The Face Recognition App is a Python-based application that utilizes the face_recognition library and OpenCV to perform real-time face recognition on a video stream. The app allows you to capture faces, associate them with names, and recognize those faces in subsequent frames of the video.

# Features
Real-time face detection and recognition using the webcam
Load pre-existing face encodings from images to recognize known faces
Add new faces to the recognition database during runtime
Display bounding boxes and names of recognized faces in the video stream
Simple and intuitive graphical user interface

# Notes
The captured faces are not saved permanently. They are stored in memory during the (runtime) of the application.
The application uses the face_recognition library, which provides an easy-to-use interface for face detection and recognition.
The graphical user interface (GUI) is built using the Tkinter library, which is included in the standard Python distribution.
To improve the feature and save the captured faces for future use, you can modify the code to store the face encodings and associated names in a file or a database. 
