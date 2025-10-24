import cv2
import os
from ultralytics import YOLO
import argparse

def crop_main_object(image_path, output_path, model):
    """
    Detects objects in an image, finds the largest 'book', and crops the image to that object.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return

        results = model(image)

        book_detections = []
        # The model.names attribute is a dict mapping class_id to class_name
        # e.g. {0: 'person', 1: 'bicycle', ..., 73: 'book'}
        # We need to find the class_id for 'book'.
        book_class_id = -1
        for class_id, class_name in model.names.items():
            if class_name == 'book':
                book_class_id = class_id
                break
        
        if book_class_id == -1:
            print("Error: 'book' class not found in the model.")
            return

        for result in results:
            for box in result.boxes:
                if int(box.cls) == book_class_id:
                    book_detections.append(box.xyxy[0].tolist()) # .tolist() to convert tensor to list

        if not book_detections:
            print(f"No book or sheet detected in {image_path}")
            return

        # Find the largest detected book by area
        largest_book = max(book_detections, key=lambda det: (det[2] - det[0]) * (det[3] - det[1]))
        
        x1, y1, x2, y2 = map(int, largest_book)

        cropped_image = image[y1:y2, x1:x2]
        
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved to {output_path}")

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the main object (answer sheet or book) from an image.")
    parser.add_argument("image_files", nargs='*', default=['D:/data/OCR/all_jpg/1447692.jpg'], help="Path to one or more image files. Defaults to D:/data/OCR/all_jpg/1447692.jpg if not provided.")
    
    args = parser.parse_args()

    # Load a pre-trained YOLOv8 model
    # The model will be downloaded automatically on the first run.
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt') 
    print("YOLO model loaded.")

    for image_file in args.image_files:
        if not os.path.isfile(image_file):
            print(f"Error: File not found at {image_file}")
            continue

        filename, ext = os.path.splitext(image_file)
        output_filename = f"{filename}_cropped{ext}"
        
        crop_main_object(image_file, output_filename, model)
