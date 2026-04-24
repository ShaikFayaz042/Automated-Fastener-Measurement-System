# Measure/washer.py
import cv2
import numpy as np
from pathlib import Path
import os

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- Helper Functions ----------------
def detect_inner_diameter(image, center, outer_radius):
    """Detect inner circle using Canny + HoughCircles inside outer radius."""
    mask_circle = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask_circle, (int(center[0]), int(center[1])), int(outer_radius * 0.9), 255, -1)
    roi = cv2.bitwise_and(image, image, mask=mask_circle)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1, minDist=outer_radius / 2,
        param1=50, param2=30,
        minRadius=int(outer_radius * 0.1),
        maxRadius=int(outer_radius * 0.7)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Pick circle closest to center
        best_circle = None
        min_distance = float("inf")
        for cx, cy, r in circles[0]:
            dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            if dist < min_distance:
                min_distance = dist
                best_circle = (cx, cy, r)
        if best_circle is not None:
            return 2 * best_circle[2]  # ID in pixels
    return None


def refine_outer_contour(mask):
    """Clean mask and extract largest contour as washer outer edge."""
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    outer_contour = max(contours, key=cv2.contourArea)
    (x, y), outer_radius = cv2.minEnclosingCircle(outer_contour)
    return (x, y), outer_radius


# ---------------- Main Measurement ----------------
def measure(det, mask_data, px_per_mm, save_dir=None):
    """
    Measure washer OD & ID.

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
    image = cv2.imread(det.get("image_path", ""))  # if available
    if image is None:
        print("⚠️ No original image path inside det; cannot draw overlay.")
        return {}

    roi_img = image[y1:y2, x1:x2]
    roi_mask = mask_data[y1:y2, x1:x2]

    if roi_img.size == 0 or roi_mask.size == 0:
        return {}

    # Get outer + inner diameters
    center, outer_radius = refine_outer_contour(roi_mask)
    if center is None:
        return {"status": "❌ Could not detect washer contour"}

    OD_px = 2 * outer_radius
    ID_px = detect_inner_diameter(roi_img, center, outer_radius)

    # Convert to mm
    OD_mm = OD_px / px_per_mm
    ID_mm = ID_px / px_per_mm if ID_px else None

    # ---------------- Visualization on full image ----------------
    result_img = image.copy()
    # Offset center to full image coordinates
    center_full = (int(center[0] + x1), int(center[1] + y1))
    cv2.circle(result_img, center_full, int(outer_radius), (0, 255, 0), 2)
    if ID_px:
        cv2.circle(result_img, center_full, int(ID_px / 2), (0, 0, 255), 2)

    # Put text near the object (top-left of bbox)
    text_x, text_y = x1 + 10, y1 + 30
    cv2.putText(result_img, f"OD: {OD_mm:.2f}mm", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if ID_mm:
        cv2.putText(result_img, f"ID: {ID_mm:.2f}mm", (text_x, text_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save annotated full image
    out_img_path = Path(save_dir) / f"{det['class_name']}_{det.get('id', 'x')}_measured.jpg"
    cv2.imwrite(str(out_img_path), result_img)


    # Save dimensions to txt (include class for Model D)
    out_txt_path = Path(save_dir) / f"{det['class_name']}_{det.get('id', 'x')}_measured.txt"
    with open(out_txt_path, "w") as f:
        f.write("class: washer\n")
        f.write(f"OD_mm: {OD_mm:.2f}\n")
        if ID_mm:
            f.write(f"ID_mm: {ID_mm:.2f}\n")
        f.write(f"px_per_mm: {px_per_mm:.4f}\n")


    return {
        "class": "washer",
        "OD_mm": OD_mm,
        "ID_mm": ID_mm,
        "px_per_mm": px_per_mm,
        "image_saved": str(out_img_path),
        "txt_saved": str(out_txt_path)
    }
