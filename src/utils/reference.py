import cv2
import numpy as np
import os

def order_points(pts):
    """Order contour points as TL, TR, BR, BL consistently."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def detect_reference(image_path, ref_size_mm=20.0, save_path=None):
    """
    Detects a near-square reference object in the image and calculates px/mm.

    Args:
        image_path (str): Path to the input image.
        ref_size_mm (float): Real-world size of reference square side in mm.
        save_path (str): Folder to save annotated image. If None, no image saved.

    Returns:
        tuple: (status, px_per_mm, ref_square_points)
               status = 'success' or 'failed'
    """
    img = cv2.imread(image_path)
    if img is None:
        return "failed", None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (better for uneven lighting)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 10
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_square = None
    px_per_mm = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # skip noise
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:  # quadrilateral
            box = approx.reshape(-1, 2)
            (tl, tr, br, bl) = order_points(box)
            width = np.linalg.norm(tr - tl)
            height = np.linalg.norm(bl - tl)
            aspect_ratio = min(width, height) / max(width, height)

            if aspect_ratio > 0.9:  # near-square
                ref_square = np.array([tl, tr, br, bl], dtype=np.int32)
                side_px = (width + height) / 2.0
                px_per_mm = side_px / ref_size_mm
                break

    if ref_square is not None:
        cv2.drawContours(img, [ref_square], -1, (0, 255, 0), 3)
        cx = int(np.mean(ref_square[:, 0]))
        cy = int(np.mean(ref_square[:, 1]))
        cv2.putText(img, f"Ref {ref_size_mm}mm", (cx - 60, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            out_file = os.path.join(save_path, os.path.basename(image_path))
            cv2.imwrite(out_file, img)

        return "success", px_per_mm, ref_square
    else:
        return "failed", None, None

# ---------------- Test run ----------------
if __name__ == "__main__":
    status, px_per_mm, pts = detect_reference(
        r"Results\Detection\capture_1757686922_0_annotated.jpg",
        ref_size_mm=20.0,
        save_path=r"Results\Reference"
    )
