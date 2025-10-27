from ultralytics import YOLO
import torch
from torchsummary import summary
import cv2
import os


def save_image(image, path):
    """
    Save an image to a specified path.

    Parameters:
        image (ndarray): The image to be saved.
        path (str): The path where the image will be saved.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the image
    cv2.imwrite(path, image)


model = YOLO("D:/Projects/OCR-answer-sheet/runs/segment/train11/weights/best.pt")  # load a pretrained model (recommended for training)
img = 'D:/data/OCR/all_jpg/1447353.jpg'  # or file, Path, PIL, OpenCV, numpy, list
image = cv2.imread(img)
# new_width = 720
# new_height = 720
# # Resize the image
# image = cv2.resize(image, (new_width, new_height))
# Inference
img = 'D:/Projects/OCR-answer-sheet/garbage/image1.jpg'

save_image(image, img)
results = model.predict(img, save=True, imgsz=640, conf=0.3, save_txt=True, show=False)
print(results[0].boxes.data) # returns xyxy of bounding box + confidence and class number