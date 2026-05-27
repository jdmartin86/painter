import os
import cv2
import numpy as np

# Load the Haar Cascade classifier once when the module is imported
# This avoids the overhead of reloading the XML file on every single frame
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configuration matching your server layout
IMAGE_W = 128
IMAGE_H = 128

# Build an absolute path to the ONNX file
BASE_DIR = "/Users/jdmartin/Documents/Code/painter/server/"
MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

# Initialize YuNet deep-learning detector
# Score threshold: 0.6 means 60% confidence required to count as a face
_YUNET = cv2.FaceDetectorYN.create(
    model=MODEL_PATH,
    config="",
    input_size=(IMAGE_W, IMAGE_H),
    score_threshold=0.6,
    nms_threshold=0.3,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

def detect_face_dnn(frame: np.ndarray) -> int:
    """
    Advanced deep-learning face detection using YuNet.
    Extremely accurate at long distances and low resolutions.
    """
    if frame is None:
        return 0
        
    try:
        # YuNet expects a standard 3-channel image matrix layout (H, W, 3)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            working_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 2:
            working_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            working_frame = frame

        # Ensure frame dimensions match YuNet initialized size profile
        if working_frame.shape[1] != IMAGE_W or working_frame.shape[0] != IMAGE_H:
            working_frame = cv2.resize(working_frame, (IMAGE_W, IMAGE_H))

        # Perform inference
        retval, faces = _YUNET.detect(working_frame)
        
        # If faces are found, faces will be a numpy array. If none, it returns None.
        if faces is not None and len(faces) > 0:
            return 1
            
        return 0

    except Exception as e:
        print(f"[face_detector] YuNet evaluation error: {e}")
        return 0

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
    return float(detect_face_dnn(frame))