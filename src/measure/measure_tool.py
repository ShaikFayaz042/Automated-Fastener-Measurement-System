# Measure/measure_tool.py
"""
Model C (Measurement Tool)

This module decides which measurement function to run
for each detected object type (bolt, washer, nut, screw).
It connects with main.py, taking inputs from Model A (detections),
Model B (masks), and reference (px_per_mm).
"""

try:
    from . import bolt, washer, nut, screw
except ImportError:
    import bolt, washer, nut, screw

def process_measurements(image_path, detections, label_path, mask_data, px_per_mm):
    """
    Run measurement pipeline (Model C).

    Args:
        image_path (str): Path to original image
        detections (list): YOLO detections (from Model A)
        label_path (str): YOLO label file (from Model A)
        mask_data (dict/np.array): segmentation masks (from Model B)
        px_per_mm (float): scale factor from reference square

    Returns:
        dict: Results for each object with measurements
    """
    results = {}

    if px_per_mm is None:
        print("⚠️ No reference scale found. Cannot compute dimensions.")
        return results

    for i, det in enumerate(detections):
        cls_name = det["class_name"].lower()
        obj_id = f"{cls_name}_{i+1}"

        # add extra info into detection dict
        det["image_path"] = image_path
        det["id"] = i + 1

        # --- dispatch by class ---
        if cls_name == "bolt":
            results[obj_id] = bolt.measure(det, mask_data, px_per_mm)

        elif cls_name == "washer":
            results[obj_id] = washer.measure(det, mask_data, px_per_mm)

        elif cls_name == "nut":
            results[obj_id] = nut.measure(det, mask_data, px_per_mm)

        elif cls_name == "screw":
            results[obj_id] = screw.measure(det, mask_data, px_per_mm)

        else:
            results[obj_id] = {"status": f"❌ No measurement module for {cls_name}"}

    return results


# ----------------- Test Run -----------------
if __name__ == "__main__":
    dummy_detections = [
        {"class_name": "washer", "xyxy": (200, 250, 280, 320)},
    ]

    # Simulated call (with dummy mask array)
    import numpy as np
    dummy_mask = np.zeros((500, 500), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.circle(dummy_mask, (240, 285), 30, 255, -1)

    results = process_measurements(
        image_path="test.jpg",
        detections=dummy_detections,
        label_path="test.txt",
        mask_data=dummy_mask,
        px_per_mm=3.2
    )

    print("✅ Measurement results:")
    for obj, data in results.items():
        print(obj, "->", data)
