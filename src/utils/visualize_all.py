# utils/visualize_all.py

import cv2
import numpy as np

def check_collision(box1, box2):
    """Check if two boxes overlap"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

def find_non_overlapping_position(bg_box, existing_boxes, image_shape, bbox):
    """Find a position for info box that doesn't overlap with existing boxes"""
    x1, y1, x2, y2 = bbox
    bg_w = bg_box[2] - bg_box[0]
    bg_h = bg_box[3] - bg_box[1]
    
    # Try different positions: right, left, top, bottom, top-right, bottom-right, etc.
    positions = [
        (x2 + 10, y1, "right"),  # Right side
        (x1 - bg_w - 10, y1, "left"),  # Left side
        (x1, y1 - bg_h - 10, "top"),  # Top
        (x1, y2 + 10, "bottom"),  # Bottom
        (x2 + 10, y1 - bg_h - 10, "top-right"),  # Top-right
        (x2 + 10, y2 + 10, "bottom-right"),  # Bottom-right
        (x1 - bg_w - 10, y1 - bg_h - 10, "top-left"),  # Top-left
        (x1 - bg_w - 10, y2 + 10, "bottom-left"),  # Bottom-left
    ]
    
    for pos_x, pos_y, side in positions:
        # Ensure within image bounds
        if pos_x < 10:
            pos_x = 10
        if pos_x + bg_w > image_shape[1] - 10:
            pos_x = image_shape[1] - bg_w - 10
        if pos_y < 10:
            pos_y = 10
        if pos_y + bg_h > image_shape[0] - 10:
            pos_y = image_shape[0] - bg_h - 10
        
        new_box = (pos_x, pos_y, pos_x + bg_w, pos_y + bg_h)
        
        # Check collision with existing boxes
        collision = False
        for existing_box in existing_boxes:
            if check_collision(new_box, existing_box):
                collision = True
                break
        
        if not collision:
            return (pos_x, pos_y), side
    
    # If all positions collide, return the right side position anyway (will draw line)
    return (x2 + 10, y1), "right"

def visualize_detections(image, detections, masks, measurements):
    """
    Draws bbox (white), mask (light red transparent), and dimensions (dark green).
    Handles both full-processing components and detection-only components.
    Implements collision detection to prevent overlapping info boxes.
    Draws connecting lines if info box is far from object.

    detections: list of dicts [{ 'label': 'Nut', 'bbox': [x1,y1,x2,y2] }, ...]
    masks: list of binary mask arrays (same size as image, dtype=uint8)
    measurements: list of dicts [{ 'AF': 10.1, 'ID': 9.1, 'OD': 12.2, 'Nominal_M': 'M6', 'Length_mm': 20.0 }, ...]
    """

    overlay = image.copy()
    
    # First pass: collect all info box positions to avoid overlaps
    info_boxes = []  # List of (bg_x1, bg_y1, bg_x2, bg_y2, text_x, text_y, dim_texts, bbox, side)
    
    for i, det in enumerate(detections):
        label = det['label']
        x1, y1, x2, y2 = map(int, det['bbox'])
        dims = measurements[i] if i < len(measurements) else {}

        if dims:  # Only process components with measurements
            # --- 5. Dimension Text (right side of bbox, dark green) for measured components ---
            dim_texts = []
            if label.lower() == "nut":
                # Display as Dia and AF separately
                if "Nominal_Dia" in dims and dims["Nominal_Dia"]:
                    dim_texts.append(f"Dia = {dims['Nominal_Dia']}")
                if "AF" in dims:
                    dim_texts.append(f"AF = {dims['AF']:.0f} mm")
                # Fallback to measured values if no predicted values
                if not dim_texts:
                    if "AF" in dims:
                        dim_texts.append(f"AF = {dims['AF']:.2f} mm")
            elif label.lower() == "washer":
                if "OD" in dims:
                    dim_texts.append(f"OD = {dims['OD']:.2f} mm")
                if "ID" in dims:
                    dim_texts.append(f"ID = {dims['ID']:.2f} mm")
            elif label.lower() == "bolt":
                # Display as Bolt Size and Length separately
                if "Nominal_M" in dims and dims["Nominal_M"]:
                    dim_texts.append(f"Bolt Size = {dims['Nominal_M']}")
                if "Length_mm" in dims:
                    dim_texts.append(f"Length = {dims['Length_mm']:.1f} mm")
                # Fallback to measured values if no predicted values
                if not dim_texts:
                    if "Width_mm" in dims:
                        dim_texts.append(f"Dia = {dims['Width_mm']:.2f} mm")
                    if "Length_mm" in dims:
                        dim_texts.append(f"L = {dims['Length_mm']:.2f} mm")
            elif label.lower() == "screw":
                # Display predicted dimensions in the format requested
                if "Nominal_Dia" in dims and dims["Nominal_Dia"]:
                    dim_texts.append(f"Nominal Dia = {dims['Nominal_Dia']}mm")
                if "Length_mm" in dims:
                    dim_texts.append(f"Length = {dims['Length_mm']:.0f}mm")
                # Fallback to measured values if no predicted values
                if not dim_texts:
                    if "Head_Dia_mm" in dims:
                        dim_texts.append(f"Head Dia = {dims['Head_Dia_mm']:.2f} mm")
                    if "Length_mm" in dims:
                        dim_texts.append(f"L = {dims['Length_mm']:.2f} mm")

            # Calculate text dimensions for positioning
            if dim_texts:
                max_text_width = 0
                total_text_height = len(dim_texts) * 25
                
                for txt in dim_texts:
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    max_text_width = max(max_text_width, tw)
                
                # Calculate background box dimensions
                bg_w = max_text_width + 10
                bg_h = total_text_height + 10
                
                # Find non-overlapping position
                existing_boxes = [box[:4] for box in info_boxes]  # Get existing box coordinates
                (text_x, text_y), side = find_non_overlapping_position(
                    (0, 0, bg_w, bg_h), existing_boxes, image.shape, (x1, y1, x2, y2)
                )
                
                # Calculate background box position
                bg_x1 = text_x - 5
                bg_y1 = text_y - 5
                bg_x2 = text_x + max_text_width + 5
                bg_y2 = text_y + total_text_height + 5
                
                # Store info box data for second pass
                info_boxes.append((bg_x1, bg_y1, bg_x2, bg_y2, text_x, text_y, dim_texts, (x1, y1, x2, y2), side))
    
    # Second pass: Draw all elements (masks, bounding boxes, labels, and info boxes)
    for i, det in enumerate(detections):
        label = det['label']
        x1, y1, x2, y2 = map(int, det['bbox'])
        dims = measurements[i] if i < len(measurements) else {}

        # --- 1. Draw Mask (light red transparent) - only for components with masks ---
        if i < len(masks) and masks[i].max() > 0:  # Only draw mask if it's not empty
            mask = masks[i].astype(np.uint8)
            if mask.max() > 1:  # normalize if mask is 0/255
                mask = (mask > 127).astype(np.uint8)

            # Create colored mask (red)
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[:, :, 2] = mask * 255

            # Blend only where mask==1
            overlay = np.where(mask[:, :, None].astype(bool),
                               cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0),
                               overlay)

        # --- 2. Draw Bounding Box (white) ---
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # --- 3. Object Label (centered for all objects, top-left for others) ---
        label_text = f"{label}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Top-left for other objects
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 6, y1), (50, 50, 50), -1)
        cv2.putText(overlay, label_text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- 4. Handle Detection-Only Components ---
        if not dims:  # Empty measurements means detection-only component
            # Add "Detection Only" text
            cv2.putText(overlay, "Detection Only", (x1 + 3, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw info boxes and connecting lines
    for i, (bg_x1, bg_y1, bg_x2, bg_y2, text_x, text_y, dim_texts, bbox, side) in enumerate(info_boxes):
        x1, y1, x2, y2 = bbox
        
        # Calculate distance from info box to object
        box_center_x = (bg_x1 + bg_x2) // 2
        box_center_y = (bg_y1 + bg_y2) // 2
        obj_center_x = (x1 + x2) // 2
        obj_center_y = (y1 + y2) // 2
        
        distance = np.sqrt((box_center_x - obj_center_x)**2 + (box_center_y - obj_center_y)**2)
        
        # Draw connecting line if box is far from object (more than 50 pixels)
        if distance > 50:
            # Determine connection points
            if side == "right":
                line_start = (x2, (y1 + y2) // 2)
                line_end = (bg_x1, box_center_y)
            elif side == "left":
                line_start = (x1, (y1 + y2) // 2)
                line_end = (bg_x2, box_center_y)
            elif side == "top":
                line_start = ((x1 + x2) // 2, y1)
                line_end = (box_center_x, bg_y2)
            elif side == "bottom":
                line_start = ((x1 + x2) // 2, y2)
                line_end = (box_center_x, bg_y1)
            else:
                # For diagonal positions, connect to nearest corner
                line_start = (x2, y1) if "right" in side else (x1, y1)
                line_end = (box_center_x, box_center_y)
            
            # Draw dashed line
            cv2.line(overlay, line_start, line_end, (0, 255, 0), 2)
            # Draw small circle at connection point on object
            cv2.circle(overlay, line_start, 3, (0, 255, 0), -1)
            cv2.circle(overlay, line_end, 3, (0, 255, 0), -1)
        
        # Draw background rectangle for info box
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 150, 0), 2)
        
        # Draw text
        for j, txt in enumerate(dim_texts):
            text_y_pos = text_y + 25 + j * 25
            cv2.putText(overlay, txt, (text_x, text_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return overlay
