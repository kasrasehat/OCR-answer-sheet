import os
import argparse
import cv2
from augment import Augmentor

def augment_images_in_folder(images_folder, backgrounds_folder, output_folder):
    """
    Applies the scale_rotate_background augmentation to all images in a folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    augmentor = Augmentor(backgrounds_path=backgrounds_folder)
    
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

    for filename in image_files:
        image_path = os.path.join(images_folder, filename)
        
        try:
            augmented_image, angle, transport, scale = augmentor.scale_rotate_background(image_path)
            
            output_filename = f"{filename}"
            output_path = os.path.join(output_folder, output_filename)
            
            cv2.imwrite(output_path, augmented_image)
            
            print(f"Processed {filename}:")
            print(f"  - Saved to {output_path}")
            print(f"  - Angle: {angle}, Transport: {transport}, Scale: {scale}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment images by scaling, rotating, and placing them on random backgrounds.")
    
    parser.add_argument(
        "--images_folder", 
        default="D:/data/OCR/test", 
        help="Path to the folder containing images to augment."
    )
    parser.add_argument(
        "--backgrounds_folder", 
        default="D:/data/background", 
        help="Path to the folder containing background images."
    )
    parser.add_argument(
        "--output_folder", 
        default="D:/data/OCR/test_output", 
        help="Path to the folder where augmented images will be saved."
    )

    args = parser.parse_args()

    augment_images_in_folder(args.images_folder, args.backgrounds_folder, args.output_folder)
