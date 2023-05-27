import cv2
import tkinter as tk
from PIL import Image, ImageTk
from face_recognition_engine import FacialRecognizer
import os


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        self.sfr = FacialRecognizer()
       
        self.sfr.load_encoding_images("images/")
        self.sfr.load_encoding_data()

        self.cap = cv2.VideoCapture(0)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.name_entry = tk.Entry(root)
        self.name_entry.pack()

        self.capture_button = tk.Button(root, text="Capture", command=self.capture_face)
        self.capture_button.pack()

        self.quit_button = tk.Button(root, text="Quit", command=self.quit)
        self.quit_button.pack()

        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()

        face_locations, face_names = self.sfr.detect_known_faces(frame)

        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2, cv2.LINE_AA)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        self.video_label.configure(image=img)
        self.video_label.image = img

        self.root.after(30, self.update_video)

    def capture_face(self):
        ret, frame = self.cap.read()
        cv2.imshow("Capture", frame)
        name = self.name_entry.get()

        self.sfr.add_encoding(frame, name)

        self.name_entry.delete(0, tk.END)

    def quit(self):
        self.sfr.save_encoding_data()
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


root = tk.Tk()
app = FaceRecognitionApp(root)
root.mainloop()
