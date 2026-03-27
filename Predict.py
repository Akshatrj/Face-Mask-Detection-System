import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


model_path = os.path.join(os.path.dirname(__file__), "mask_detector.pth")
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    print("Please run train.py first to train and save the model.")
    exit(1)

model=CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_face(face):
    face=cv2.resize(face,(64,64)).astype(np.float32)/255.0
    face=torch.tensor(face,dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    return face


def predict_face(face):
    output=model(face)
    _, pred=torch.max(output, 1)

    if pred.item()==0:
        return "Mask",(0,255,0)
    else:
        return "No Mask",(0,0,255)


def test_image(path):
    img=cv2.imread(path)
    if img is None:
        print("Image not found")
        return

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, 1.1, 7, minSize=(80, 80))

    for (x,y, w,h) in faces:
        face=preprocess_face(img[y:y+h,x:x+w])
        text,color=predict_face(face)

        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__=="__main__":
    while True:
        print("\n======MASK DETECTOR======")
        print("1-Test Image")
        print("2-Exit")

        choice=input("Enter your choice: ")

        if choice=="1":
            root=tk.Tk()
            root.withdraw()
            path=filedialog.askopenfilename(
                title="Select an Image",
                filetypes=[("Image Files","*.png *.jpg *.jpeg *.bmp *.webp"),("All Files","*.*")]
            )
            root.destroy()
            if path:
                test_image(path)
            else:
                print("No image selected.")

        elif choice=="2":
            print("Exiting")
            break

        else:
            print("Invalid choice")
