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


model = YOLO("/home/kasra/PycharmProjects/YOLOv8_customize/runs/segment/predict_val/weights/best.pt")  # load a pretrained model (recommended for training)
img = '/home/kasra/kasra_files/data-shenasname/ai_files_20230606/0010049681_0.jpg'  # or file, Path, PIL, OpenCV, numpy, list
image = cv2.imread(img)
new_width = 720
new_height = 720
# Resize the image
image = cv2.resize(image, (new_width, new_height))
# Inference
img = '/home/kasra/PycharmProjects/YOLOv8_customize/extra_files/image1.jpg'

save_image(image, img)
results = model.predict(img, save=True, imgsz=640, conf=0.3, save_txt=True, show=False)
print(results[0].boxes.data) # returns xyxy of bounding box + confidence and class number