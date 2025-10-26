# OCR-answer-sheet

Tools and scripts to preprocess, augment, detect, segment, and crop answer-sheet components using YOLOv8.

## Quick start (Windows, Python 3.12)

1) Clone and create a virtual environment

```powershell
cd D:\Projects\OCR-answer-sheet
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install base dependencies

```powershell
pip install -r requirements.txt
```

3) Choose your PyTorch build
- CPU only (works everywhere):
  ```powershell
  pip uninstall -y torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```
- NVIDIA GPU with CUDA 12.4 (recommended if you have a CUDA-capable GPU):
  ```powershell
  pip uninstall -y torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
- Verify:
  ```powershell
  python - << 'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
PY
  ```

4) Adjust paths
This repo uses absolute Windows paths (e.g., `D:/data/...`). Update paths in scripts or pass CLI arguments as needed to match your environment.

## Data layout and labels

The YOLO segmentation dataset paths and class names are configured in `config.yaml`:

```yaml
path: D:/Projects/OCR-answer-sheet/data
train: D:/Projects/OCR-answer-sheet/data/train
val: D:/Projects/OCR-answer-sheet/data/validation
names:
  0: sheet
  1: question_box
  2: exam_id
  3: student_id
```

- Ensure your images and segmentation labels (YOLOv8 format) are under `train` and `validation` folders.
- Class names are referenced by training and test scripts.

## Scripts

### 1) Convert WebP images to JPG
`convert_images.py`
- Converts all `.webp` in a source folder to `.jpg` in a destination folder.
- Defaults are in the script:
  - Source: `D:/data/OCR/hessam`
  - Destination: `D:/data/OCR/all_jpg`

Run:
```powershell
python convert_images.py
```
To change folders, edit `source_directory` and `destination_directory` at the bottom of the script.

### 2) Crop the main object (book/sheet) using YOLO
`crop_sheet.py`
- Detects the largest `book` object with YOLOv8 and crops the image to that bounding box.
- Usage:
  ```powershell
  python crop_sheet.py D:/data/OCR/all_jpg/1447692.jpg
  # or multiple images
  python crop_sheet.py D:/img/a.jpg D:/img/b.jpg
  ```
- If no image path is provided, it defaults to `D:/data/OCR/all_jpg/1447692.jpg`.

### 3) Augment images on random backgrounds
`augment.py`, `run_augmentation.py`
- `Augmentor.scale_rotate_background(image_path)` scales/rotates the image and places it on a random background, returning the augmented image with metadata.
- `run_augmentation.py` applies that to a whole folder.

Run (with defaults):
```powershell
python run_augmentation.py
```
Custom paths:
```powershell
python run_augmentation.py ^
  --images_folder "D:/data/OCR/test" ^
  --backgrounds_folder "D:/data/background" ^
  --output_folder "D:/data/OCR/test_output"
```

### 4) Train YOLOv8 segmentation
`train_segmentation.py`
- Trains a segmentation model (`yolov8n-seg.pt`) using the dataset in `config.yaml`.

GPU training:
```powershell
# Ensure CUDA build of torch is installed
python train_segmentation.py
```

CPU training (slower):
- Edit `train_segmentation.py` and change `device='cuda:0'` to `device='cpu'`.
- Optionally reduce memory use by lowering batch size and workers, e.g.: `batch=2, workers=0`.

Outputs are saved under `runs/segment/train*/`.

### 5) Segment, auto-rotate by IDs, and save crops
`test.py`
- Pipeline:
  1. First detection on the original image to locate the largest `student_id` and `exam_id` boxes.
  2. Compute the angle of the line between their centers and rotate the image by that angle (via PIL `rotate(expand=True)`) to make the line horizontal.
  3. Detect again on the rotated image. If `question_box` exists and the average of (`student_id`, `exam_id`) centers is not in its upper-left, rotate an additional 180° and re-detect.
  4. Save an oriented preview and crop each detected mask into `output_dir/<image_stem>/` as `<image_stem>_<label>_<idx>.png`.

Run with defaults:
```powershell
python test.py
```
Custom arguments:
```powershell
python test.py ^
  --weights "D:/Projects/OCR-answer-sheet/runs/segment/train11/weights/best.pt" ^
  --image   "D:/data/OCR/all_jpg/1447277.jpg" ^
  --output_dir "D:/Projects/OCR-answer-sheet/runs/segment/crops" ^
  --imgsz 640 --conf 0.3
```
Outputs per image:
- `<image_stem>_oriented_preview.jpg`
- `<image_stem>_rotation_info.txt` (applied rotation degrees)
- Crops: `<image_stem>_<label>_<idx>.png`

### 6) Simple segmentation test (optional)
`test_segmentation.py`
- Minimal example of running segmentation on a single image and saving predictions.

## Troubleshooting

- "Invalid CUDA device" / `torch.cuda.is_available(): False`
  - Install the CUDA-enabled PyTorch build (`cu124`) and update NVIDIA drivers, or switch `device='cpu'` in the training script.

- WinError 1455: paging file too small (e.g., when loading CUDA DLLs)
  - Increase Windows virtual memory (pagefile).
  - Close other GPU-intensive apps.
  - Reduce training memory: use a smaller model, set `batch=2`, and on Windows set `workers=0`.

- Performance tips
  - Keep `imgsz` moderate (e.g., 640) for memory.
  - Use `yolov8n` models for lower VRAM usage.

- Ultralytics version
  - If prompted to update Ultralytics, you can run:
    ```powershell
    pip install -U ultralytics
    ```

## Notes
- Large binary weights (`*.pt`, `*.pth`) are ignored by `.gitignore` to avoid bloating the repo.
- Paths in scripts are Windows-style absolute paths. Adjust or pass arguments as needed.
- Results and logs are saved under `runs/segment/...` by default.
