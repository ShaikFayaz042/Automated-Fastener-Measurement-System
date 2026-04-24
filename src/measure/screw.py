# Measure/screw.py
import cv2
import numpy as np
from pathlib import Path
import os

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def measure(det, mask_data, px_per_mm, save_dir=None):
    """
    Measure screw length and diameter using minAreaRect.
    
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
    
    # Find contours in mask
    contours, _ = cv2.findContours(roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "❌ Could not detect screw contour"}
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    
    # Calculate dimensions
    width, height = rect[1]

    # For screws: use the longer side as the total screw length
    # and the shorter side as the head diameter.
    length_px = max(width, height)
    head_diameter_px = min(width, height)

    # Convert to mm
    length_mm = length_px / px_per_mm
    head_diameter_mm = head_diameter_px / px_per_mm  # Head diameter for screws
    
    # Draw rectangle on full image
    result_img = image.copy()
    # Offset box points to full image coordinates
    box_full = box + np.array([[x1, y1]])
    cv2.drawContours(result_img, [box_full], 0, (0, 0, 255), 2)
    
    # Put text near the object (top-left of bbox) - Length and Head Diameter
    text_x, text_y = x1 + 10, y1 + 30
    cv2.putText(result_img, f"L: {length_mm:.2f}mm", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_img, f"Head Dia: {head_diameter_mm:.2f}mm", (text_x, text_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Save annotated full image
    out_img_path = Path(save_dir) / f"{det['class_name']}_{det.get('id', 'x')}_measured.jpg"
    cv2.imwrite(str(out_img_path), result_img)
    
    # Save dimensions to txt (include class for Model D)
    out_txt_path = Path(save_dir) / f"{det['class_name']}_{det.get('id', 'x')}_measured.txt"
    with open(out_txt_path, "w") as f:
        f.write("class: screw\n")
        f.write(f"Length_mm: {length_mm:.2f}\n")
        f.write(f"Head_Dia_mm: {head_diameter_mm:.2f}\n")
        f.write(f"px_per_mm: {px_per_mm:.4f}\n")
    
    
    return {
        "class": "screw",
        "Length_mm": length_mm,
        "Head_Dia_mm": head_diameter_mm,
        "px_per_mm": px_per_mm,
        "image_saved": str(out_img_path),
        "txt_saved": str(out_txt_path)
    }
