import face_recognition
import cv2
import os
import glob
import csv
import numpy as np

class FacialRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

        # Create directory to store face encodings if it doesn't exist
        self.face_encodings_dir = "face_encodings"
        if not os.path.exists(self.face_encodings_dir):
            os.makedirs(self.face_encodings_dir)

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)

        print("Encoding images loaded")

    def load_encoding_data(self):
        data_file = os.path.join(self.face_encodings_dir, "face_data.csv")
        if not os.path.exists(data_file):
            return

        with open(data_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                name = row[0]
                encoding = np.array(row[1:], dtype=float)
                self.known_face_names.append(name)
                self.known_face_encodings.append(encoding)

    def save_encoding_data(self):
        data_file = os.path.join(self.face_encodings_dir, "face_data.csv")

        with open(data_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for name, encoding in zip(self.known_face_names, self.known_face_encodings):
                row = [name] + encoding.tolist()
                writer.writerow(row)

        print("Face encoding data saved")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

    def add_encoding(self, image, name):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_encoding = face_recognition.face_encodings(rgb_img)[0]

        self.known_face_encodings.append(img_encoding)
        self.known_face_names.append(name)
