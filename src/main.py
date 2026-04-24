import os  
import cv2
import numpy as np
from utils.capture import capture_image
from utils.reference import detect_reference
from utils.detect_objects import run_modelA
from utils.generate_masks import run_modelB
from measure import measure_tool   # 🔥 connect Model C
from utils.match_spec import run_spec_match       # 🔥 connect Model D
from utils.visualize_all import visualize_detections
from pathlib import Path

# Get project root (parent of src directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Helper function to get absolute paths
def get_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)


def main():
    print("🚀 Starting main program...")

    # Clean results/measurements and results/predictions for a fresh run
    import shutil
    for folder in [get_path("outputs/5_measured"), get_path("outputs/6_results")]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    Path(get_path("outputs/2_reference")).mkdir(parents=True, exist_ok=True)
    Path(get_path("outputs/3_detection")).mkdir(parents=True, exist_ok=True)
    Path(get_path("outputs/3_detection/labels")).mkdir(parents=True, exist_ok=True)
    Path(get_path("outputs/4_segmentation/masks")).mkdir(parents=True, exist_ok=True)
    Path(get_path("outputs/4_segmentation/overlay")).mkdir(parents=True, exist_ok=True)
    Path(get_path("outputs/5_measured")).mkdir(parents=True, exist_ok=True)
    Path(get_path("outputs/6_results/spec_match_report")).mkdir(parents=True, exist_ok=True)
    Path(get_path("outputs/6_results/output_images")).mkdir(parents=True, exist_ok=True)

    # Step 1: Capture image
    status, img_path = capture_image()
    if status != "success":
        print("⚠️ No image saved. Exiting program.")
        return

    # Step 2: Detect reference square
    ref_status, px_per_mm, ref_points = detect_reference(
        image_path=img_path,
        ref_size_mm=20.0,              # standard reference square size in mm
        save_path=get_path("outputs/2_reference")  # folder to save annotated reference image
    )

    if ref_status == "success":
        print(f"✅ Reference detected. Pixels per mm: {px_per_mm:.2f}")
    else:
        print("❌ Reference not detected. Check image quality or lighting.")
        px_per_mm = None

    # Step 3: Run Model A (Object Detection)
    print("🎯 Running Model A for object detection...")
    detections, label_path = run_modelA(
        image_path=img_path,
        model_path=get_path("models/model_a.pt"),
        device="cpu",       # change to "0" for GPU
        imgsz=640,
        conf_thr=0.25,
        iou_thr=0.5,
        save_annotated=True,
        outdir=get_path("outputs/3_detection"),
        save_labels=True
    )

    if detections:
        print("\n✅ Detected objects:")
        for i, det in enumerate(detections):
            print(f"{i+1}. {det['class_name']}")
        print()
    else:
        print("⚠️ No objects detected.")

    # Step 4: Run Model B (Segmentation/Mask Generation)
    mask_array = None
    if label_path and detections:
        print("🎯 Running Model B for segmentation/mask generation...")
        result = run_modelB(
            img_path=img_path,
            label_txt_path=label_path
        )

        if result and "mask_array" in result:
            mask_array = result["mask_array"]
            print("✅ Mask generated and saved successfully.")
            print()
        else:
            print("⚠️ No masks generated.")
    else:
        print("⚠️ Skipping Model B (no labels from Model A).")

    # Step 5: Run Model C (Measurements)
    measurement_results = {}
    if detections:
        print("📏 Running Model C (Measurements)...")
        measurement_results = measure_tool.process_measurements(
            image_path=img_path,
            detections=detections,
            label_path=label_path,
            mask_data=mask_array,
            px_per_mm=px_per_mm
        )

        if measurement_results:
            for obj_id, res in measurement_results.items():
                class_name = res.get('class', 'unknown')
                if class_name == 'washer':
                    print("✅ Washer measurement saved.")
                elif class_name == 'bolt':
                    print("✅ Bolt measurement saved.")
                elif class_name == 'nut':
                    print("✅ Nut measurement saved.")
                elif class_name == 'screw':
                    print("✅ Screw measurement saved.")
        else:
            print("⚠️ No valid measurements returned.")
    else:
        print("⚠️ Skipping Model C (no detections).")

    # Step 6: Run Model D (Specification Matching)
    spec_results = []
    if measurement_results:
        print("📊 Running Model D (Specification Matching)...")
        report_path = get_path("outputs/6_results/spec_match_report/spec_match_report.txt")
        reference_csv_dict = {
            "washer": get_path("data/datasets/washers_dataset.csv"),
            "bolt": get_path("data/datasets/bolts_dataset.csv"),
            "nut": get_path("data/datasets/nuts_dataset.csv"),
            "screw": get_path("data/datasets/screws_dataset.csv")
        }
        spec_results = run_spec_match(
            measurements_dir=get_path("outputs/5_measured"),
            reference_csv_dict=reference_csv_dict,
            output_txt=report_path
        )
        print(f"✅ Spec matching completed. Report saved at: {report_path}")
    else:
        print("⚠️ Skipping Model D (no measurements to match).")

    # --- Visualization: overlay bbox, mask, predicted dims ---
    if detections:
        image = cv2.imread(img_path)
        det_list = []
        mask_list = []
        meas_list = []
        
        # Create a mapping of object_id to predicted values
        predicted_values = {}
        for spec_res in spec_results:
            if spec_res['reference']:
                file_name = Path(spec_res['file']).stem
                obj_id = file_name.replace('_measured', '')
                predicted_values[obj_id] = spec_res['reference']
        
        # Process all detections for visualization
        for idx, det in enumerate(detections):
            cls_name = det['class_name'].lower()
            bbox = det['xyxy']
            
            det_list.append({'label': det['class_name'], 'bbox': bbox})
            
            # Find corresponding measurement result
            obj_id = f"{cls_name}_{idx+1}"
            meas = measurement_results.get(obj_id, {})
            
            # Extract mask from Model B
            if mask_array is not None:
                x1, y1, x2, y2 = map(int, bbox)
                if isinstance(mask_array, (list, tuple)) and idx < len(mask_array):
                    obj_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    mask_obj = mask_array[idx]
                    if mask_obj.shape != obj_mask.shape:
                        mask_obj = cv2.resize(mask_obj, (obj_mask.shape[1], obj_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    obj_mask = (mask_obj > 0).astype(np.uint8)
                    mask_list.append(obj_mask)
                elif isinstance(mask_array, np.ndarray) and mask_array.ndim == 3 and idx < mask_array.shape[0]:
                    mask_obj = mask_array[idx]
                    obj_mask = (mask_obj > 0).astype(np.uint8)
                    mask_list.append(obj_mask)
                elif isinstance(mask_array, np.ndarray) and mask_array.ndim == 2:
                    obj_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    mask_crop = mask_array[y1:y2, x1:x2]
                    obj_mask[y1:y2, x1:x2] = (mask_crop > 0).astype(np.uint8)
                    mask_list.append(obj_mask)
                else:
                    mask_list.append(np.zeros(image.shape[:2], dtype=np.uint8))
            else:
                mask_list.append(np.zeros(image.shape[:2], dtype=np.uint8))
            
            # Use predicted values if available, otherwise fall back to measured values
            pred_vals = predicted_values.get(obj_id, {})
            dims = {}
            
            if meas.get('class') == 'bolt' and pred_vals:
                if 'Bolt Size' in pred_vals:
                    dims['Nominal_M'] = pred_vals['Bolt Size']
                if 'Length_mm' in pred_vals and pred_vals['Length_mm'] is not None:
                    dims['Length_mm'] = float(pred_vals['Length_mm'])
            elif meas.get('class') == 'washer' and pred_vals:
                if 'OD_mm' in pred_vals and pred_vals['OD_mm'] is not None:
                    dims['OD'] = float(pred_vals['OD_mm'])
                if 'ID_mm' in pred_vals and pred_vals['ID_mm'] is not None:
                    dims['ID'] = float(pred_vals['ID_mm'])
            elif meas.get('class') == 'nut' and pred_vals:
                if 'Nominal Dia' in pred_vals:
                    dims['Nominal_Dia'] = pred_vals['Nominal Dia']
                if 'AF_mm' in pred_vals and pred_vals['AF_mm'] is not None:
                    dims['AF'] = float(pred_vals['AF_mm'])
            elif meas.get('class') == 'screw' and pred_vals:
                if 'Length_mm' in pred_vals and pred_vals['Length_mm'] is not None:
                    dims['Length_mm'] = float(pred_vals['Length_mm'])
                if 'Nominal Dia' in pred_vals:
                    dims['Nominal_Dia'] = pred_vals['Nominal Dia']
            else:
                if meas.get('OD_mm') is not None:
                    dims['OD'] = float(meas['OD_mm'])
                if meas.get('ID_mm') is not None:
                    dims['ID'] = float(meas['ID_mm'])
                if meas.get('AF_mm') is not None:
                    dims['AF'] = float(meas['AF_mm'])
                if meas.get('Length_mm') is not None:
                    dims['Length_mm'] = float(meas['Length_mm'])
            
            if meas.get('class') == 'screw' and 'Length_mm' in meas and 'Length_mm' not in dims and meas['Length_mm'] is not None:
                dims['Length_mm'] = float(meas['Length_mm'])
            
            meas_list.append(dims)
        
        # Visualize all detections
        if det_list:
            result_img = visualize_detections(image, det_list, mask_list, meas_list)
            out_img_path = get_path("outputs/6_results/output_images/final_output.jpg")
            cv2.imwrite(out_img_path, result_img)
            
            print("\n🖼️ Displaying final output image...")
            cv2.imshow("Final Detection & Measurement Results", result_img)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print("\n📏 PROCESSING SUMMARY:")
            print(f"   Components with measurements: {len(measurement_results)}")
            
            for obj_id, meas in measurement_results.items():
                print(f"\n   🔧 {obj_id.upper()}:")
                print(f"      Class: {meas['class']}")
                
                if meas['class'] == 'bolt':
                    pred = predicted_values.get(obj_id, {})
                    print(f"      ➤ Bolt Size: {pred.get('Bolt Size', 'N/A')}")
                    print(f"      ➤ AF (Across Flats): {pred.get('AF_mm', 'N/A')} mm")
                    print(f"      ➤ Length: {pred.get('Length_mm', 'N/A')} mm")
                        
                elif meas['class'] == 'washer':
                    pred = predicted_values.get(obj_id, {})
                    print(f"      ➤ Outer Diameter (OD): {pred.get('OD_mm', 'N/A')} mm")
                    print(f"      ➤ Inner Diameter (ID): {pred.get('ID_mm', 'N/A')} mm")
                        
                elif meas['class'] == 'nut':
                    pred = predicted_values.get(obj_id, {})
                    print(f"      ➤ Nut Size: {pred.get('Nut Size', 'N/A')}")
                    print(f"      ➤ Across Flats (AF): {pred.get('AF_mm', 'N/A')} mm")
                        
                elif meas['class'] == 'screw':
                    pred = predicted_values.get(obj_id, {})
                    print(f"      ➤ Screw Size: {pred.get('Screw Size', 'N/A')}")
                    print(f"      ➤ Length: {pred.get('Length_mm', 'N/A')} mm")
            
            print("\n✅ ANALYSIS COMPLETED...")
            print("run again for next components")

if __name__ == "__main__":
    main()
