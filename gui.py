# Import required Libraries
import tkinter
from tkinter import *
from PIL import Image, ImageTk
import cv2
from cringe_detector import classify_image

# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window
win.geometry("700x350")

# Create a Label to capture the Video frames
label = Label(win)
label.grid(row=0, column=0)
cap= cv2.VideoCapture(0)
cringe_rating = "This image is x% cringe"
cringe_label = Label(win, textvariable=cringe_rating)


# Define function to show frame
def show_frames():
    # Get the latest frame and convert into Image
    cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    button = tkinter.Button(win, command=take_capture)
    button.grid(row=1, column=0)

    label.after(20, show_frames)


def take_capture():
    global cringe_rating
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    result = classify_image(img)
    cringe_rating = f"This image is {result}% cringe"


show_frames()
win.mainloop()
