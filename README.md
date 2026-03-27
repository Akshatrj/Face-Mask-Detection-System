# 😷 Face Mask Detection System

Face mask detection system is a system that detects whether the person in image is wearing mask or not.Its a machine learing model based on deep learning using Custom trained CNN Model built with libraries like **Pytorch** and **OpenCV**.

## 1. Project Overview

A deep learning-based tool that takes an image as input, detects faces using OpenCV's Haar Cascade, and classifies each face as "Mask" or "No Mask" using a custom-trained CNN built in PyTorch.

## 2. Problem Statement
Wearing mask is very important to prevent the spread of any type of contagious disease. And we saw this during covid-19 pandemic. Manually checking every person at entrances to hospitals, offices, or campuses is impractical. so here a basic system that detects if a person is wearing mask or not with option of future growth to optimize it for real time detection using webcam.

## 3. Objectives

- The Program accepts an image as input via file dialog  
- It Detect faces in the image using Haar Cascade  
- Checks if person is wearing mask or not
- Display the result with color-coded bounding boxes  
- Keep the project modular and easy to understand  

## 4. Technologies Used
### PyTorch
Used to build and train custom CNN Model

### OpenCV
Used for face detection and image processing.

### NumPy
Used for array manipulation and image data preprocessing.

### Matplotlib
Used to make training accuracy/loss plots and confusion matrix.

### Scikit-learn
Used for train/test split, classification report,confusion matrix.

### Tkinter
Used to open file dialog box to select image.

### Kaggle Dataset Link-
```
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
```

## 5. Features
- Custom CNN trained from scratch. 
- Supports all standard image formats (PNG, JPG, JPEG, BMP, WebP)  
- Color-coded output: 🟩 Green = Mask, 🟥 Red = No Mask  
- Terminal based menu to select option.
- Displays training metrics and plots.

## 6. How to Run
```
1. Install Python 3.8 or above 
```
``` 
2. Install required packages:
```
      pip install torch torchvision opencv-python numpy scikit-learn matplotlib

3. Prepare dataset in data/ folder:
```
data/
├── with_mask/
└── without_mask/

```
4. Train the model:
```
First run train.py
```
5. Run predictions:
```
Run Predict.py to test on images.

```


## 7. Project Structure

```
Face Mask Detection System/
├── data/
│   ├── with_mask/          # Images of people wearing masks
│   └── without_mask/       # Images of people without masks
├── train.py                # Training script
├── Predict.py              # Prediction script
├── requirements.txt        # Python dependencies
├── mask_detector.pth       # Trained model weights (generated after training)
├── Project_Report.md       # Detailed project report
└── README.md               # This file
```

## 8. Project Workflow
```
Step 1: Train the Model
               Run train.py

Step 2: Run Predictions
               Run Predict.py

Option 1 - Test Image
               Select an image via file dialog
               Faces are detected and classified
               Result displayed with bounding boxes

Option 2 - Exit
```

## 9. Algorithm

**Step-1-** Load the dataset containing images of faces with and without masks.  
**Step-2-** Train the Model.  
**Step-3-** Input the Image.  
**Step-4-** A face is detected in the input image using OpenCV's Haar Cascade classifier.  
**Step-5-** The detected face region is cropped, resized to 64×64 pixels, and normalized.  
**Step-6-** The preprocessed face is passed through a trained CNN model.  
**Step-7-** The model outputs a prediction: **Mask** (With Face in green box) or **No Mask** (With Face in red box).  
**Step-8-** The result is drawn on the original image and displayed.

## 🧑‍💻 Author
**Akshat Rajput**


