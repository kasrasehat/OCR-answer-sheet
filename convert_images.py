import os
from PIL import Image

def convert_webp_to_jpg(source_dir, dest_dir):
    """
    Converts all .webp images in the source directory to .jpg format
    and saves them in the destination directory.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".webp"):
            source_path = os.path.join(source_dir, filename)
            dest_filename = os.path.splitext(filename)[0] + ".jpg"
            dest_path = os.path.join(dest_dir, dest_filename)

            try:
                with Image.open(source_path) as img:
                    img.convert("RGB").save(dest_path, "jpeg")
                print(f"Converted {filename} to {dest_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    source_directory = r"D:\data\OCR\hessam"
    destination_directory = r"D:\data\OCR\all_jpg"
    convert_webp_to_jpg(source_directory, destination_directory)
