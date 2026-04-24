from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import os

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- USER CONFIG ----------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/model_a.pt")       # YOLOv8 model path
DEVICE = "cpu"                          # or "0" for GPU
IMG_SIZE = 640
CONF_THR = 0.25
IOU_THR = 0.5
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/3_detection")       # annotated images folder
LABEL_DIR = os.path.join(PROJECT_ROOT, "outputs/3_detection/labels")           # save labels folder
# --------------------------------------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def xyxy_to_xywh(xyxy):
    """Convert (x1,y1,x2,y2) to (cx,cy,w,h)"""
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    return cx, cy, w, h

def non_max_suppression(boxes, scores, iou_thresh=0.5):
    """Simple NMS to remove overlapping boxes"""
    if len(boxes) == 0:
        return []
    idxs = cv2.dnn.NMSBoxes(
        bboxes=[[int(x1), int(y1), int(x2-x1), int(y2-y1)] for (x1,y1,x2,y2) in boxes],
        scores=[float(s) for s in scores],
        score_threshold=0.0,
        nms_threshold=iou_thresh
    )
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()

def save_labels_yolo_format(image_path, hbb_list, label_dir=LABEL_DIR):
    """Saves labels in YOLO TXT format and returns path"""
    ensure_dir(label_dir)
    stem = Path(image_path).stem
    label_path = Path(label_dir) / f"{stem}.txt"

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    with open(label_path, "w") as f:
        for det in hbb_list:
            cls_id = det["class_id"]
            cx, cy, bw, bh = det["xywh"]
            cxn, cyn, w_n, h_n = cx/w, cy/h, bw/w, bh/h
            f.write(f"{cls_id} {cxn:.6f} {cyn:.6f} {w_n:.6f} {h_n:.6f}\n")
    return str(label_path)

def run_modelA(image_path, model_path=MODEL_PATH, device=DEVICE, imgsz=IMG_SIZE,
               conf_thr=CONF_THR, save_annotated=True, outdir=OUTPUT_DIR, 
               iou_thr=IOU_THR, save_labels=True):
    """
    Runs YOLOv8 HBB detection on image and optionally saves labels.
    
    Returns:
        hbb_list (list of dicts): Each detection info
        label_path (str or None): Path to saved label file
    """
    ensure_dir(outdir)
    ensure_dir(LABEL_DIR)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"❌ Could not read image: {image_path}")
        return [], None

    # ⚠️ No resizing here, image is already 640×640 from capture.py

    model = YOLO(model_path)
    results = model.predict(source=img_bgr, device=device, imgsz=imgsz, conf=conf_thr, verbose=False)
    result = results[0]

    try:
        names = model.names
    except Exception:
        names = {i: str(i) for i in range(1000)}

    hbb_list = []
    label_path = None

    if len(result.boxes) > 0:
        xyxy_all = result.boxes.xyxy.cpu().numpy()
        conf_all = result.boxes.conf.cpu().numpy()
        cls_all  = result.boxes.cls.cpu().numpy().astype(int)

        keep_idxs = non_max_suppression(xyxy_all, conf_all, iou_thresh=iou_thr)

        for i in keep_idxs:
            x1, y1, x2, y2 = xyxy_all[i]
            cx, cy, w_box, h_box = xyxy_to_xywh((x1, y1, x2, y2))
            conf = float(conf_all[i])
            cls_id = int(cls_all[i])
            cls_name = names.get(cls_id, str(cls_id))

            entry = {
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": conf,
                "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                "xywh": (float(cx), float(cy), float(w_box), float(h_box))
            }
            hbb_list.append(entry)

    # Annotate image
    if save_annotated and len(hbb_list) > 0:
        annotated = result.plot()
        outpath = Path(outdir) / (Path(image_path).stem + "_annotated.jpg")
        cv2.imwrite(str(outpath), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # Save labels
    if save_labels and len(hbb_list) > 0:
        label_path = save_labels_yolo_format(image_path, hbb_list, label_dir=LABEL_DIR)

    return hbb_list, label_path

# ----------------- Test run -----------------
if __name__ == "__main__":
    IMAGE_PATH = "outputs/1_captured_images/capture_1757686922_0.jpg"  # replace with your test image
    detections, label_path = run_modelA(IMAGE_PATH)
    if detections:
        print("\nDetected objects:")
        for i, det in enumerate(detections):
            print(f"{i+1}. {det['class_name']} conf={det['conf']:.3f}, xyxy={tuple(round(v,2) for v in det['xyxy'])}")
        print(f"\nLabels saved at: {label_path}")
    else:
        print("No objects detected.")
