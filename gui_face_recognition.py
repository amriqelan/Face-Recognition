import cv2
import tkinter as tk
from PIL import Image, ImageTk
from face_recognition_engine import FacialRecognizer


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        # Create instance of FacialRecognizer
        self.sfr = FacialRecognizer()
        self.sfr.load_encoding_images("images/")

        # Create video capture
        self.cap = cv2.VideoCapture(0)

        # Create GUI elements
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.name_entry = tk.Entry(root)
        self.name_entry.pack()

        self.capture_button = tk.Button(root, text="Capture", command=self.capture_face)
        self.capture_button.pack()

        self.quit_button = tk.Button(root, text="Quit", command=self.quit)
        self.quit_button.pack()

        # Start video stream
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()

        # Detect faces
        face_locations, face_names = self.sfr.detect_known_faces(frame)

        # Draw rectangles and names on the frame
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)  # Green rectangle
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2, cv2.LINE_AA)

        # Convert the frame to ImageTk format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        # Update the video label
        self.video_label.configure(image=img)
        self.video_label.image = img

        # Repeat the update after a delay (30 milliseconds)
        self.root.after(30, self.update_video)

    def capture_face(self):
        # Capture an image from the video feed
        ret, frame = self.cap.read()

        # Display the captured image
        cv2.imshow("Capture", frame)

        # Prompt the user to enter a name for the captured face
        name = self.name_entry.get()

        # Save the captured face to the recognition database
        self.sfr.add_encoding(frame, name)

        # Clear the name entry field
        self.name_entry.delete(0, tk.END)

    def quit(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


# Create the Tkinter root window
root = tk.Tk()

# Create an instance of the FaceRecognitionApp
app = FaceRecognitionApp(root)

# Start the Tkinter event loop
root.mainloop()