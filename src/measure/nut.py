# Measure/nut.py
import cv2
import numpy as np
from pathlib import Path
import os

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- Helper Functions ----------------
def detect_nut_af(mask):
    """
    Detect nut AF (across flats) using rotated rectangle method.
    The shorter side of the minimum area rectangle represents the across flats dimension.
    """
    mask_uint8 = (mask * 255).astype("uint8")
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    # Find the largest contour
    outer_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle (rotated rectangle)
    rect = cv2.minAreaRect(outer_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Calculate side lengths of the rotated rectangle
    side1 = np.linalg.norm(box[0] - box[1])
    side2 = np.linalg.norm(box[1] - box[2])
    
    # The shorter side is the across flats dimension for a hex nut
    AF_px = min(side1, side2)
    
    return AF_px, box

# ---------------- Main Measurement ----------------
def measure(det, mask_data, px_per_mm, save_dir=None):
    """
    Measure nut AF (across flats) using minAreaRect method.

    Args:
        det (dict): detection entry from Model A
        mask_data (np.array): full mask (binary)
        px_per_mm (float): scale factor from reference
        save_dir (str): folder to save outputs

    Returns:
        dict: measurement results
    """
    # Use default path if not provided
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "outputs/5_measured")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Crop ROI from mask + image (using bbox from Model A)
    x1, y1, x2, y2 = map(int, det["xyxy"])
    image = cv2.imread(det.get("image_path", ""))
    if image is None:
        print("⚠️ No original image path inside det; cannot draw overlay.")
        return {}

    roi_img = image[y1:y2, x1:x2]
    roi_mask = mask_data[y1:y2, x1:x2]

    if roi_img.size == 0 or roi_mask.size == 0:
        return {}

    # Detect nut AF using rotated rectangle
    AF_px, rotated_rect = detect_nut_af(roi_mask)
    
    if AF_px is None:
        return {"status": "❌ Could not detect nut AF"}

    # Convert to mm
    AF_mm = AF_px / px_per_mm

    # Draw rectangle on full image
    result_img = image.copy()
    if rotated_rect is not None:
        # Offset box points to full image coordinates
        box_full = rotated_rect + np.array([[x1, y1]])
        cv2.drawContours(result_img, [box_full], 0, (0, 255, 0), 2)

    # Put text near the object (top-left of bbox)
    text_x, text_y = x1 + 10, y1 + 30
    cv2.putText(result_img, f"AF: {AF_mm:.2f}mm", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save annotated full image
    out_img_path = Path(save_dir) / f"{det['class_name']}_{det.get('id', 'x')}_measured.jpg"
    cv2.imwrite(str(out_img_path), result_img)

    # Save dimensions to txt (include class for Model D)
    out_txt_path = Path(save_dir) / f"{det['class_name']}_{det.get('id', 'x')}_measured.txt"
    with open(out_txt_path, "w") as f:
        f.write("class: nut\n")
        f.write(f"AF_mm: {AF_mm:.2f}\n")
        f.write(f"px_per_mm: {px_per_mm:.4f}\n")


    return {
        "class": "nut",
        "AF_mm": AF_mm,
        "px_per_mm": px_per_mm,
        "image_saved": str(out_img_path),
        "txt_saved": str(out_txt_path)
    }
