import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_image(image: np.ndarray, path: str) -> None:
    ensure_dir(path)
    cv2.imwrite(path, image)


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Run segmentation, rotate image using ID centers, ensure UL placement, then crop each mask.")
    parser.add_argument(
        "--weights",
        default="D:/Projects/OCR-answer-sheet/runs/segment/train11/weights/best.pt",
        help="Path to trained segmentation weights.",
    )
    parser.add_argument(
        "--image",
        default="D:/data/OCR/all_jpg/1447277.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/Projects/OCR-answer-sheet/runs/segment/crops",
        help="Directory to save cropped segments.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold.",
    )

    args = parser.parse_args()

    # Load model and image
    model = YOLO(args.weights)
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    image_stem = os.path.splitext(os.path.basename(args.image))[0]
    image_output_dir = os.path.join(args.output_dir, image_stem)
    os.makedirs(image_output_dir, exist_ok=True)

    # First pass: detect on original image to compute rotation
    res0 = model.predict(image, save=False, imgsz=args.imgsz, conf=args.conf, save_txt=False, show=False)[0]
    boxes0 = res0.boxes
    names = res0.names if hasattr(res0, 'names') else {}

    def pick_center(res_boxes, label_name):
        best = None
        best_area = -1.0
        for i in range(len(res_boxes)):
            cls_id = int(res_boxes.cls[i].item()) if res_boxes is not None and len(res_boxes) > i else -1
            cls_name = names.get(cls_id, f'class_{cls_id}')
            if cls_name != label_name:
                continue
            x1, y1, x2, y2 = res_boxes.xyxy[i].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        return best

    student_c = pick_center(boxes0, 'student_id')
    exam_c = pick_center(boxes0, 'exam_id')

    total_angle_deg = 0.0
    rotated_cv = image.copy()

    if student_c is not None and exam_c is not None:
        vx = exam_c[0] - student_c[0]
        vy = exam_c[1] - student_c[1]
        theta_deg = float(np.degrees(np.arctan2(vy, vx)))  # y-down coordinates
        # Rotate by -theta using PIL with expand=True
        pil_img = cv2_to_pil(image)
        pil_rot = pil_img.rotate(theta_deg, expand=True, resample=Image.BILINEAR)
        rotated_cv = pil_to_cv2(pil_rot)
        total_angle_deg += theta_deg

    # Second pass: detect on rotated image
    res1 = model.predict(rotated_cv, save=False, imgsz=args.imgsz, conf=args.conf, save_txt=False, show=False)[0]
    boxes1 = res1.boxes
    names1 = res1.names if hasattr(res1, 'names') else {}

    def pick_center_boxes(res_boxes, names_map, label_name):
        best = None
        best_area = -1.0
        for i in range(len(res_boxes)):
            cls_id = int(res_boxes.cls[i].item()) if res_boxes is not None and len(res_boxes) > i else -1
            cls_name = names_map.get(cls_id, f'class_{cls_id}')
            if cls_name != label_name:
                continue
            x1, y1, x2, y2 = res_boxes.xyxy[i].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        return best

    s1 = pick_center_boxes(boxes1, names1, 'student_id')
    e1 = pick_center_boxes(boxes1, names1, 'exam_id')
    q1 = pick_center_boxes(boxes1, names1, 'question_box')

    # If question_box exists, ensure avg(S,E) is upper-left; else optional extra 180°
    if s1 is not None and e1 is not None and q1 is not None:
        ids_cx = (s1[0] + e1[0]) / 2.0
        ids_cy = (s1[1] + e1[1]) / 2.0
        qx, qy = q1
        if not (ids_cx < qx and ids_cy < qy):
            pil_rot = cv2_to_pil(rotated_cv)
            pil_rot2 = pil_rot.rotate(180, expand=True, resample=Image.BILINEAR)
            rotated_cv = pil_to_cv2(pil_rot2)
            total_angle_deg += 180.0
            # Re-detect after 180
            res1 = model.predict(rotated_cv, save=False, imgsz=args.imgsz, conf=args.conf, save_txt=False, show=False)[0]

    # Save oriented preview and rotation info
    oriented_path = os.path.join(image_output_dir, f"{image_stem}_oriented_preview.jpg")
    save_image(rotated_cv, oriented_path)
    rot_info_path = os.path.join(image_output_dir, f"{image_stem}_rotation_info.txt")
    with open(rot_info_path, 'w', encoding='utf-8') as f:
        f.write(f"total_rotation_applied_deg: {total_angle_deg:.2f}\n")

    # Use final detection (res1) for crops
    res = res1
    orig = res.orig_img
    masks = res.masks
    boxes = res.boxes
    names = res.names if hasattr(res, 'names') else {}

    if masks is None or masks.data is None or len(masks.data) == 0:
        print("No segments detected on rotated image.")
        return

    H, W = orig.shape[:2]

    for i in range(len(masks.data)):
        # get binary mask resized to original image size
        m = masks.data[i].cpu().numpy()  # (h, w) float 0..1
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m_bin = (m > 0.5).astype(np.uint8) * 255

        # apply mask to original image
        seg = cv2.bitwise_and(orig, orig, mask=m_bin)

        # compute tight crop bounds
        ys, xs = np.where(m_bin > 0)
        if ys.size and xs.size:
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            seg_c = seg[y1:y2, x1:x2]
        else:
            seg_c = seg

        # class name
        cls_id = int(boxes.cls[i].item()) if boxes is not None and len(boxes) > i else -1
        cls_name = names.get(cls_id, f'class_{cls_id}')

        # build filename and save
        out_path = os.path.join(image_output_dir, f"{image_stem}_{cls_name}_{i}.png")
        save_image(seg_c, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
