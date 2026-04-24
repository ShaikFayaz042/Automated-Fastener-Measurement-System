from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------- Load Model B (segmentation) ----------
SEG_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/model_b.pt")
seg_model = YOLO(SEG_MODEL_PATH)  # trained seg weight


# --------- Helper: Read YOLOv8 HBB label file ----------
def read_yolo_hbb_labels(txt_path, img_shape):
    """
    Reads YOLO HBB format: class x_center y_center width height (normalized)
    Returns list of bbox coords: [x1,y1,x2,y2,class_id]
    """
    h, w = img_shape[:2]
    bboxes = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_c, y_c, bw, bh = map(float, parts)
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            bboxes.append({"bbox": [x1, y1, x2, y2], "class": int(cls)})
    return bboxes


# --------- Auto-crop Segmentation Function ----------
def run_modelB(img_path, label_txt_path, seg_model=seg_model,
               conf=0.25, min_area=50,
               masks_dir=None, images_dir=None):
    
    # Use default paths if not provided
    if masks_dir is None:
        masks_dir = os.path.join(PROJECT_ROOT, "outputs/4_segmentation/masks")
    if images_dir is None:
        images_dir = os.path.join(PROJECT_ROOT, "outputs/4_segmentation/overlay")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    bboxes = read_yolo_hbb_labels(label_txt_path, img.shape)

    final_mask = np.zeros((h, w), dtype=np.uint8)

    for det in bboxes:
        x1, y1, x2, y2 = det["bbox"]

        # --- Crop ROI from image ---
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # --- Run segmentation on cropped ROI ---
        results = seg_model.predict(crop, imgsz=640, conf=conf, verbose=False)
        mask_crop = None

        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255
            if masks.ndim == 3 and masks.shape[0] > 0:
                # take the largest mask
                areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
                idx = int(np.argmax(areas))
                mask_crop = masks[idx]

        # --- fallback if mask not found ---
        if mask_crop is None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, mask_crop = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- Safety: force mask size match with crop ---
        if mask_crop.shape != (y2 - y1, x2 - x1):
            mask_crop = cv2.resize(mask_crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        # --- Place mask back into full-size canvas ---
        final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], mask_crop)

    # -------- Morphological cleanup --------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    # Remove small objects
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    clean_mask = np.zeros_like(final_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 255

    # -------- Overlay for visualization --------
    overlay = img.copy()
    colored_mask = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(overlay, 0.75, colored_mask, 0.25, 0)

    # -------- Save results --------
    Path(masks_dir).mkdir(parents=True, exist_ok=True)
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    base = Path(img_path).stem
    cv2.imwrite(str(Path(images_dir) / f"{base}_seg_overlay.jpg"), overlay)
    cv2.imwrite(str(Path(masks_dir) / f"{base}_mask.png"), clean_mask)
    np.save(str(Path(masks_dir) / f"{base}_mask.npy"), clean_mask)

    return {"masked_image": overlay, "mask_array": clean_mask}


# --------- Test run ----------
if __name__ == "__main__":
    image_path = "outputs/3_detection/capture_1757686922_0_annotated.jpg"
    label_path = "outputs/3_detection/labels/capture_1757686922_0.txt"  # YOLO HBB format label file

    result = run_modelB(image_path, label_path)
    print("✅ Segmentation complete, results saved in outputs/4_segmentation/masks & outputs/4_segmentation/overlay.")
