import cv2
import os
import time
import numpy as np

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAVE_DIR = os.path.join(PROJECT_ROOT, "outputs/1_captured_images")
os.makedirs(SAVE_DIR, exist_ok=True)

def resize_image(image, target_size=(640, 640)):
    """Resize while maintaining aspect ratio and padding with black."""
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def capture_image(camera_index=1):
    """
    Opens camera with live feed.
    Controls:
        c -> capture image (preview only)
        k -> confirm last captured image, save & exit
        r -> retake (discard last capture, continue live feed)
        q -> quit without saving

    Returns (status, filepath) where status = 'success' or 'failed'.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Cannot access camera.")
        return "failed", None

    last_frame = None
    img_counter = 0
    print("🎥 Controls: [c] capture | [k] confirm/save | [r] retake | [q] quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        cv2.imshow("Live Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Capture (but not save yet)
            last_frame = frame.copy()
            preview = resize_image(last_frame, (640, 640))
            cv2.imshow("Captured Preview", preview)
            print("📸 Preview captured. Press [k] to confirm/save or [r] to retake.")

        elif key == ord('k') and last_frame is not None:  # Confirm + save
            timestamp = int(time.time())
            img_name = f"capture_{timestamp}_{img_counter}.jpg"
            img_path = os.path.join(SAVE_DIR, img_name)

            # Always save resized image
            resized = resize_image(last_frame, (640, 640))
            cv2.imwrite(img_path, resized)
            print("🎉 Image saved.")

            cap.release()
            cv2.destroyAllWindows()
            return "success", img_path

        elif key == ord('r'):  # Retake
            last_frame = None
            cv2.destroyWindow("Captured Preview")
            print("🔄 Retake: Preview cleared, continue live feed.")

        elif key == ord('q'):  # Quit
            print("🚪 Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return "failed", None


# Test run
if __name__ == "__main__":
    status, path = capture_image()
    print("Result:", status, path)
