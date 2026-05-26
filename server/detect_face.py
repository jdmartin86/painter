import cv2
import numpy as np

# Load the Haar Cascade classifier once when the module is imported
# This avoids the overhead of reloading the XML file on every single frame
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(frame: np.ndarray) -> int:
    """
    Detects if a face is present in the given numpy frame.
    Outputs 1 if a face is detected, otherwise 0.
    
    Accepts grayscale matrices, standard BGR images, or frames with an 
    explicit tracking channel dimension (H, W, 1).
    """
    if frame is None:
        return 0
        
    try:
        # 1. Normalize input layout for OpenCV's Cascade Classifier (requires 2D Grayscale)
        if frame.ndim == 3:
            if frame.shape[-1] == 1:
                # Squeeze out tracking channel: (H, W, 1) -> (H, W)
                working_frame = frame.squeeze(axis=-1)
            elif frame.shape[-1] == 3:
                # Convert standard BGR to Grayscale if a 3-channel image is passed
                working_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                working_frame = frame
        else:
            working_frame = frame

        # 2. Detect faces
        # scaleFactor: Specifies how much the image size is reduced at each image scale.
        # minNeighbors: Higher values detect fewer faces but with higher quality.
        faces = _FACE_CASCADE.detectMultiScale(
            working_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # 3. Return 1 if the face list is not empty, else 0
        return 1 if len(faces) > 0 else 0

    except Exception as e:
        print(f"[face_detector] Error evaluating frame: {e}")
        return 0
    
def compute_reward(frame: np.ndarray) -> float:
    return detect_face(frame)