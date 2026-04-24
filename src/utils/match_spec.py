from pathlib import Path
import csv
import os
from typing import Dict, Tuple, List, Any

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============== BOLT STANDARDS (ISO 4017) ==============
BOLT_STANDARDS: Dict[str, Dict[str, Any]] = {
    'M3': {'diameter': 3.0, 'preferred_lengths': [6, 8, 10, 12, 16, 20, 25, 30]},
    'M4': {'diameter': 4.0, 'preferred_lengths': [8, 10, 12, 16, 20, 25, 30, 35, 40]},
    'M5': {'diameter': 5.0, 'preferred_lengths': [10, 12, 16, 20, 25, 30, 35, 40, 45, 50]},
    'M6': {'diameter': 6.0, 'preferred_lengths': [12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60]},
    'M8': {'diameter': 8.0, 'preferred_lengths': [16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]},
    'M10': {'diameter': 10.0, 'preferred_lengths': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100]},
    'M12': {'diameter': 12.0, 'preferred_lengths': [24, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120]},
    'M16': {'diameter': 16.0, 'preferred_lengths': [32, 40, 50, 60, 70, 80, 100, 120, 150]},
    'M20': {'diameter': 20.0, 'preferred_lengths': [40, 50, 60, 70, 80, 100, 120, 150]},
    'M24': {'diameter': 24.0, 'preferred_lengths': [48, 60, 70, 80, 100, 120, 150]},
}

# ---------------- Helper Functions ----------------
def read_measurement_txt(file_path):
    """Read a measurement txt file and return dictionary of values."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':')
                try:
                    data[key.strip()] = float(val.strip())
                except:
                    data[key.strip()] = val.strip()
    return data


def read_reference_csv(csv_path, obj_class):
    """Read reference CSV for a specific class and return a list of dicts."""
    ref_list = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if obj_class == "washer":
                ref_list.append({
                    "Nominal Dia": row["Nominal Dia (mm)"],
                    "Bolt Dia": float(row["Bolt Dia (mm)"]),
                    "ID_mm": float(row["Inner Dia (mm)"]),
                    "OD_mm": float(row["Outer Dia (mm)"]),
                    "Thickness_mm": float(row["Thickness (mm)"])
                })
            elif obj_class == "bolt":
                # Store Across Corners (mm) as AC_mm for matching
                # Length_mm is measured from image and compared with dataset
                ref_list.append({
                    "Nominal Dia": row["Nominal Dia (mm)"],
                    "Bolt Size": row["Bolt Size"],
                    "Length_mm": float(row["Length (mm)"]),
                    "AF_mm": float(row["Across Flats (mm)"]),
                    "AC_mm": float(row["Across Corners (mm)"]),
                    "Head_Thickness_mm": float(row["Head Thickness (mm)"]),
                    "Fine_Pitch_mm": float(row["Standard Fine Pitch (mm)"]),
                    "Rough_Pitch_mm": float(row["Standard Rough Pitch (mm)"])
                })
            elif obj_class == "nut":
                # Update keys as per your nuts CSV
                ref_list.append({
                    "Nominal Dia": row["Nominal Dia (mm)"],
                    "Nut Size": row["Nut Size"],
                    "AF_mm": float(row["Across Flats (mm)"]),
                    "AC_mm": float(row["Across Corners (mm)"]),
                    "Thickness_mm": float(row["Thickness (mm)"])
                })
            elif obj_class == "screw":
                # Update keys as per your screws CSV
                ref_list.append({
                    "Nominal Dia": row["Nominal Dia (mm)"],
                    "Screw Size": row["Screw Size"],
                    "Length_mm": float(row["Length (mm)"]),
                    "Diameter_mm": float(row["Shank Dia (mm)"]),
                    "Head_Dia_mm": float(row["Head Dia (mm)" ]),
                    "Thread_Pitch_mm": float(row["Thread Pitch (mm)" ]),
                    "Head_Thickness_mm": float(row["Head Thickness (mm)"])
                })
    return ref_list


def calculate_bolt_length_range(diameter: float) -> Tuple[float, float]:
    """
    Calculate min and max standard lengths for a bolt diameter per ISO 4017.
    
    Min Length = 2 × diameter
    Max Length = min(10 × diameter, 150mm)
    
    Args:
        diameter (float): Bolt diameter in mm (e.g., 6.0 for M6)
        
    Returns:
        Tuple[float, float]: (min_length_mm, max_length_mm)
        
    Example:
        >>> calculate_bolt_length_range(6.0)
        (12.0, 60.0)
        >>> calculate_bolt_length_range(24.0)
        (48.0, 150.0)
    """
    try:
        if diameter <= 0:
            return (0.0, 0.0)
        
        min_length: float = 2 * diameter
        max_length: float = min(10 * diameter, 150.0)
        
        return (min_length, max_length)
    except (TypeError, ValueError) as e:
        print(f"⚠️ Error calculating bolt length range: {e}")
        return (0.0, 0.0)


def snap_to_standard_length(measured_length: float,
                            min_length: float,
                            max_length: float,
                            preferred_lengths: List[int]) -> Dict[str, Any]:
    """
    Find the closest standard length within the valid range per ISO 4017.
    
    Args:
        measured_length (float): Measured bolt length in mm
        min_length (float): Minimum allowed length in mm
        max_length (float): Maximum allowed length in mm
        preferred_lengths (List[int]): List of preferred standard lengths
        
    Returns:
        Dict containing:
            - standard_length (int): Snapped standard length
            - deviation (float): Difference from measured
            - valid (bool): Length within valid range
            - in_range_standards (List[int]): Available standards within range
            
    Example:
        >>> snap_to_standard_length(18.78, 12, 60, [12,16,20,25,30,35,40,45,50,55,60])
        {'standard_length': 20, 'deviation': 1.22, 'valid': True, ...}
    """
    try:
        # Validate inputs
        if not isinstance(preferred_lengths, (list, tuple)) or len(preferred_lengths) == 0:
            return {
                'standard_length': None,
                'deviation': None,
                'valid': False,
                'error': 'Empty or invalid preferred_lengths list'
            }
        
        if measured_length < 0 or min_length < 0 or max_length < 0:
            return {
                'standard_length': None,
                'deviation': None,
                'valid': False,
                'error': 'Negative length values'
            }
        
        # Filter standards within valid range
        in_range_standards: List[int] = [
            length for length in preferred_lengths 
            if min_length <= length <= max_length
        ]
        
        # If no standards in range, return closest boundary
        if len(in_range_standards) == 0:
            if measured_length < min_length:
                standard_length = min_length
            else:
                standard_length = max_length
            
            return {
                'standard_length': int(standard_length),
                'deviation': float(measured_length - standard_length),
                'valid': False,
                'in_range_standards': in_range_standards,
                'error': 'Measured length outside valid range'
            }
        
        # Find closest standard length
        closest_standard = min(
            in_range_standards,
            key=lambda x: abs(float(x) - measured_length)
        )
        
        deviation: float = measured_length - closest_standard
        is_valid: bool = min_length <= measured_length <= max_length
        
        return {
            'standard_length': int(closest_standard),
            'deviation': float(round(deviation, 2)),
            'valid': is_valid,
            'in_range_standards': in_range_standards,
            'error': None
        }
        
    except (TypeError, ValueError, AttributeError) as e:
        print(f"⚠️ Error snapping to standard length: {e}")
        return {
            'standard_length': None,
            'deviation': None,
            'valid': False,
            'error': str(e)
        }


def compare_with_reference(measured, ref_list, tolerance=0.5):
    """
    Compare measured values with reference list. 
    Returns best match (min deviation) and status.
    """
    best_match = None
    min_total_dev = float('inf')

    # Special logic for bolt: match measured AC_mm to AC_mm (Across Corners)
    if ref_list and 'AC_mm' in ref_list[0] and 'AC_mm' in measured:
        for ref in ref_list:
            dev = abs(measured['AC_mm'] - ref['AC_mm'])
            if dev < min_total_dev:
                min_total_dev = dev
                best_match = ref
        
        # If best match found, apply ISO 4017 standard validation
        if best_match:
            # Step 1: Get bolt size from best match
            bolt_size = best_match.get('Bolt Size', '')
            
            # Step 2: Calculate ISO standard range for this bolt size
            diameter = best_match.get('Nominal Dia', 0)
            min_length, max_length = calculate_bolt_length_range(float(diameter))
            
            # Step 3: Get measured shaft length (total length - head thickness)
            head_thickness = best_match.get('Head_Thickness_mm', 0)
            measured_total_length = measured.get('Length_mm', 0)
            actual_shaft_length = measured_total_length - head_thickness
            measured['Actual_Length_mm'] = actual_shaft_length
            
            # Step 4: Snap to nearest standard length
            if bolt_size in BOLT_STANDARDS:
                preferred_lengths = BOLT_STANDARDS[bolt_size]['preferred_lengths']
            else:
                # Fallback: use generic standards if bolt size not in dict
                preferred_lengths = [int(x) for x in range(int(min_length), int(max_length) + 1, 5)]
            
            snap_result = snap_to_standard_length(
                actual_shaft_length,
                min_length,
                max_length,
                preferred_lengths
            )
            
            # Step 5: Store validation results
            if snap_result.get('error') is None:
                best_match['Standard_Length_mm'] = snap_result['standard_length']
                best_match['Length_Deviation_mm'] = snap_result['deviation']
                best_match['Length_Valid'] = snap_result['valid']
                best_match['Length_In_Range'] = snap_result['valid']
            else:
                best_match['Standard_Length_mm'] = actual_shaft_length
                best_match['Length_Deviation_mm'] = 0.0
                best_match['Length_Valid'] = False
                best_match['Length_In_Range'] = False
            
            # Store bolt standards info for reporting
            best_match['Min_Standard_Length'] = min_length
            best_match['Max_Standard_Length'] = max_length
            best_match['Preferred_Standards'] = preferred_lengths[:7]  # Show first 7 standards
            
            # Replace original Length_mm with snapped Standard_Length_mm for consistent output
            best_match['Length_mm'] = best_match['Standard_Length_mm']
        
        # Prepare deviation report
        deviation = {'AC_mm': measured['AC_mm'] - best_match['AC_mm']} if best_match else {}
        if best_match and 'Actual_Length_mm' in measured:
            deviation['Length_mm'] = measured['Actual_Length_mm'] - best_match.get('Standard_Length_mm', 0)
        
        status = 'pass' if best_match is not None else 'fail'
        return best_match, deviation, status
    
    # Special logic for screw: match measured Head_Dia_mm to Head_Dia_mm,
    # then compute actual shaft length = total length - head thickness and snap to a standard length.
    if ref_list and 'Head_Dia_mm' in ref_list[0] and 'Head_Dia_mm' in measured:
        for ref in ref_list:
            dev = abs(measured['Head_Dia_mm'] - ref['Head_Dia_mm'])
            if dev < min_total_dev:
                min_total_dev = dev
                best_match = ref

        if best_match:
            measured_total_length = measured.get('Length_mm', 0)
            head_thickness = best_match.get('Head_Thickness_mm', 0)
            actual_shaft_length = measured_total_length - head_thickness
            measured['Actual_Length_mm'] = actual_shaft_length

            # Use available lengths for this screw size from the dataset.
            screw_size = best_match.get('Screw Size')
            preferred_lengths = sorted({
                float(ref_item['Length_mm'])
                for ref_item in ref_list
                if ref_item.get('Screw Size') == screw_size
            })
            if not preferred_lengths:
                preferred_lengths = sorted({float(ref_item['Length_mm']) for ref_item in ref_list})

            if preferred_lengths:
                closest_standard = min(preferred_lengths, key=lambda x: abs(actual_shaft_length - x))
                min_length = min(preferred_lengths)
                max_length = max(preferred_lengths)
                length_valid = min_length <= actual_shaft_length <= max_length

                best_match['Standard_Length_mm'] = float(closest_standard)
                best_match['Length_Deviation_mm'] = float(round(actual_shaft_length - closest_standard, 2))
                best_match['Length_Valid'] = length_valid
                best_match['Length_In_Range'] = length_valid
                best_match['Min_Standard_Length'] = min_length
                best_match['Max_Standard_Length'] = max_length
                best_match['Preferred_Standards'] = preferred_lengths[:10]
                best_match['Length_mm'] = best_match['Standard_Length_mm']
            else:
                best_match['Standard_Length_mm'] = actual_shaft_length
                best_match['Length_Deviation_mm'] = 0.0
                best_match['Length_Valid'] = False
                best_match['Length_In_Range'] = False
                best_match['Min_Standard_Length'] = actual_shaft_length
                best_match['Max_Standard_Length'] = actual_shaft_length
                best_match['Preferred_Standards'] = []
                best_match['Length_mm'] = actual_shaft_length

        deviation = {'Head_Dia_mm': measured['Head_Dia_mm'] - best_match['Head_Dia_mm']} if best_match else {}
        if best_match and 'Actual_Length_mm' in measured:
            deviation['Length_mm'] = measured['Actual_Length_mm'] - best_match.get('Standard_Length_mm', 0)
        status = 'pass' if best_match is not None else 'fail'
        return best_match, deviation, status
    
    # Special logic for nut: match measured AF_mm to AF_mm (Across Flats)
    if ref_list and 'AF_mm' in ref_list[0] and 'AF_mm' in measured:
        for ref in ref_list:
            dev = abs(measured['AF_mm'] - ref['AF_mm'])
            if dev < min_total_dev:
                min_total_dev = dev
                best_match = ref
        deviation = {'AF_mm': measured['AF_mm'] - best_match['AF_mm']} if best_match else {}
        status = 'pass' if best_match is not None else 'fail'
        return best_match, deviation, status
    # Default: Only compare numeric keys present in both measured and reference
    for ref in ref_list:
        total_dev = 0
        compare_keys = [k for k in measured if k in ref and isinstance(measured[k], (int, float)) and isinstance(ref[k], (int, float))]
        for key in compare_keys:
            total_dev += abs(measured[key] - ref[key])
        if total_dev < min_total_dev:
            min_total_dev = total_dev
            best_match = ref
    deviation = {}
    status = 'pass' if best_match is not None else 'fail'
    if best_match is not None:
        for key in measured:
            if key in best_match and isinstance(measured[key], (int, float)) and isinstance(best_match[key], (int, float)):
                dev = measured[key] - best_match[key]
                deviation[key] = dev
            else:
                deviation[key] = None
    return best_match, deviation, status


# ---------------- Main Spec Match ----------------
def run_spec_match(measurements_dir=None,

                   reference_csv_dict=None,
                   output_txt=None,
                   tolerance=0.5):

    # Use default paths if not provided
    if measurements_dir is None:
        measurements_dir = os.path.join(PROJECT_ROOT, 'outputs/5_measured')
    if output_txt is None:
        output_txt = os.path.join(PROJECT_ROOT, 'outputs/6_results/spec_match_report/spec_match_results.txt')
    if reference_csv_dict is None:
        reference_csv_dict = {
            "washer": os.path.join(PROJECT_ROOT, "data/datasets/washers_dataset.csv"),
            "bolt": os.path.join(PROJECT_ROOT, "data/datasets/bolts_dataset.csv"),
            "nut": os.path.join(PROJECT_ROOT, "data/datasets/nuts_dataset.csv"),
            "screw": os.path.join(PROJECT_ROOT, "data/datasets/screws_dataset.csv")
        }

    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)

    # Read all reference datasets
    ref_dict = {}
    for obj_class, csv_path in reference_csv_dict.items():
        ref_dict[obj_class] = read_reference_csv(csv_path, obj_class)

    # Process all measurement txt files
    measurement_files = list(Path(measurements_dir).glob('*.txt'))
    results = []

    for meas_file in measurement_files:
        measured_data = read_measurement_txt(meas_file)
        obj_class = measured_data.get('class') or measured_data.get('type')
        obj_class = obj_class.lower() if obj_class else 'unknown'

        if obj_class not in ref_dict:
            print(f"⚠️ No reference for {obj_class} found.")
            continue

        best_ref, deviation, status = compare_with_reference(measured_data, ref_dict[obj_class], tolerance)

        # Save results to list
        results.append({
            'file': str(meas_file),
            'class': obj_class,
            'measured': measured_data,
            'reference': best_ref,
            'deviation': deviation,
            'status': status
        })


    # Write only measured and predicted (reference) values to txt
    with open(output_txt, 'w') as f:
        for res in results:
            f.write(f"File: {res['file']}\n")
            f.write(f"Class: {res['class']}\n")
            f.write(f"Measured Values:\n")
            for k, v in res['measured'].items():
                f.write(f"  {k}: {v}\n")
            
            # Special formatting for bolts with ISO 4017 standards
            if res['class'] == 'bolt' and res['reference']:
                f.write(f"\nISO 4017 Standards Validation:\n")
                ref = res['reference']
                f.write(f"  Bolt Size: {ref.get('Bolt Size', 'N/A')}\n")
                f.write(f"  Diameter: {ref.get('Nominal Dia', 'N/A')} mm\n")
                f.write(f"  Valid Range: {ref.get('Min_Standard_Length', 'N/A')} - {ref.get('Max_Standard_Length', 'N/A')} mm\n")
                
                prefs = ref.get('Preferred_Standards', [])
                if prefs:
                    f.write(f"  Preferred Standards (first 7): {prefs}\n")
                
                f.write(f"  Snapped Length: {ref.get('Standard_Length_mm', 'N/A')} mm\n")
                f.write(f"  Length Deviation: {ref.get('Length_Deviation_mm', 'N/A')} mm\n")
                f.write(f"  In Valid Range: {'PASS' if ref.get('Length_In_Range') else 'FAIL'}\n")
            
            if res['class'] == 'screw' and res['reference']:
                f.write(f"\nScrew Standards Validation:\n")
                ref = res['reference']
                f.write(f"  Screw Size: {ref.get('Screw Size', 'N/A')}\n")
                f.write(f"  Head Diameter: {ref.get('Head_Dia_mm', 'N/A')} mm\n")
                f.write(f"  Valid Length Range: {ref.get('Min_Standard_Length', 'N/A')} - {ref.get('Max_Standard_Length', 'N/A')} mm\n")
                f.write(f"  Snapped Length: {ref.get('Standard_Length_mm', 'N/A')} mm\n")
                f.write(f"  Length Deviation: {ref.get('Length_Deviation_mm', 'N/A')} mm\n")
                f.write(f"  In Valid Range: {'PASS' if ref.get('Length_In_Range') else 'FAIL'}\n")
            
            f.write(f"\nPredicted (Reference) Values:\n")
            for k, v in (res['reference'] or {}).items():
                # Skip internal standards fields from output
                if k not in ['Min_Standard_Length', 'Max_Standard_Length', 'Preferred_Standards', 
                           'Standard_Length_mm', 'Length_Deviation_mm', 'Length_Valid', 'Length_In_Range']:
                    # For bolts, replace Length_mm with Standard_Length_mm
                    if res['class'] == 'bolt' and k == 'Length_mm':
                        standard_length = res['reference'].get('Standard_Length_mm', v)
                        f.write(f"  {k}: {standard_length}\n")
                    else:
                        f.write(f"  {k}: {v}\n")
            f.write("\n")

    return results


# ---------------- Test Run ----------------
# if __name__ == "__main__":
#     results = run_spec_match(
#         measurements_dir="Results/Measurements",
#         reference_csv_dict={
#             "washer": "Datasets/washers_dataset.csv",
#             "bolt": "Datasets/bolts_dataset.csv",
#             "nut": "Datasets/nuts_dataset.csv",
#             "screw": "Datasets/screws_dataset.csv"
#         },
#         output_txt="Results/Predictions/spec_match_results.txt"
#     )
