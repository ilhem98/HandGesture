import cv2
import mediapipe as mp
import time
from tkinter import *
from PIL import Image, ImageTk

class HandDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Hand Detection App')

        # Create a label to display the camera frames
        self.image_label = Label(self.root)
        self.image_label.pack()

        # Open the camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Create the Mediapipe hand detection objects
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        # Initialize variables for calculating FPS
        self.pTime = 0
        self.cTime = 0

        # Start the main loop
        self.update()

    def update(self):
        # Read a frame from the camera
        success, img = self.cap.read()

        if success:
            # Convert the frame to RGB format and run hand detection
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            # Draw hand landmarks and calculate FPS
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        # print(id, lm)
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # if id == 4:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime

            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

            # Convert the frame to an image that can be displayed by a tkinter Label
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)

            # Update the image label in the tkinter app
            self.image_label.config(image=img)
            self.image_label.image = img

        # Call this method again after 1 millisecond
        self.root.after(1, self.update)

root = Tk()
app = HandDetectionApp(root)
root.mainloop()
