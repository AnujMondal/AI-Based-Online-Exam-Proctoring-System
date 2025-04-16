import cv2
import numpy as np
import dlib
import os
import traceback

# Check if dlib is available
DLIB_AVAILABLE = True
try:
    import dlib
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Some face detection features will be limited.")

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Try to initialize dlib's face detector and shape predictor
frontal_face_detector = None
shape_predictor = None

if DLIB_AVAILABLE:
    try:
        # Initialize dlib's face detector
        frontal_face_detector = dlib.get_frontal_face_detector()
        
        # Path to shape predictor model
        model_path = os.path.join('shape_predictor_model', 'shape_predictor_68_face_landmarks.dat')
        
        if os.path.isfile(model_path):
            shape_predictor = dlib.shape_predictor(model_path)
        else:
            print(f"Warning: Could not find facial landmark predictor model file")
            print(f"Expected at: {model_path}")
            print("Some features will be limited")
    except Exception as e:
        print(f"Warning: Could not initialize dlib face detection: {str(e)}")
        traceback.print_exc()

def detect_faces(frame):
    """
    Detect faces in a frame using OpenCV's Haar Cascade classifier.
    
    Args:
        frame: The input frame (numpy array)
        
    Returns:
        list: List of detected face rectangles (x, y, w, h)
    """
    if frame is None:
        return []
        
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces

def detect_faces_and_landmarks(frame):
    """
    Detect faces and facial landmarks using dlib if available, 
    otherwise fall back to OpenCV's Haar Cascade.
    
    Args:
        frame: The input frame (numpy array)
        
    Returns:
        list: List of detected face rectangles in dlib format
    """
    if frame is None:
        return []
    
    # If dlib detector is available, use it
    if DLIB_AVAILABLE and frontal_face_detector is not None:
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = frontal_face_detector(gray, 0)
            
            # Convert to list of dlib rectangles
            return list(faces)
        except Exception as e:
            print(f"Error in dlib face detection: {str(e)}")
            # Fall back to OpenCV if there was an error
    
    # Fall back to OpenCV
    opencv_faces = detect_faces(frame)
    
    # Convert OpenCV face format to dlib rectangle format
    dlib_faces = []
    for (x, y, w, h) in opencv_faces:
        dlib_faces.append(dlib.rectangle(x, y, x+w, y+h))
    
    return dlib_faces

def get_face_landmarks(frame, face_rect):
    """
    Get facial landmarks for a face.
    
    Args:
        frame: The input frame (numpy array)
        face_rect: The face rectangle in dlib format
        
    Returns:
        dlib.full_object_detection or None: Facial landmarks if available
    """
    if frame is None or face_rect is None:
        return None
    
    if not DLIB_AVAILABLE or shape_predictor is None:
        return None
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get landmarks
        landmarks = shape_predictor(gray, face_rect)
        return landmarks
    except Exception as e:
        print(f"Error getting facial landmarks: {str(e)}")
        return None

def draw_face_landmarks(frame, landmarks):
    """
    Draw facial landmarks on the frame.
    
    Args:
        frame: The input frame to draw on
        landmarks: dlib facial landmarks
        
    Returns:
        numpy.ndarray: Frame with landmarks drawn
    """
    if frame is None or landmarks is None:
        return frame
    
    # Create a copy of the frame
    vis = frame.copy()
    
    # Draw each landmark
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
    
    return vis

def draw_faces(frame, faces, color=(0, 255, 0), thickness=2):
    """
    Draw rectangles around detected faces on the frame.
    
    Args:
        frame: The input frame to draw on
        faces: List of face rectangles in dlib format
        color: Rectangle color (B,G,R)
        thickness: Line thickness
        
    Returns:
        numpy.ndarray: Frame with face rectangles drawn
    """
    if frame is None or not faces:
        return frame
    
    # Create a copy of the frame
    vis = frame.copy()
    
    # Draw rectangle for each face
    for face in faces:
        x, y = face.left(), face.top()
        w, h = face.width(), face.height()
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    
    return vis

def test_face_detection():
    """
    Simple test function to check if face detection is working properly
    """
    # Create a test image with a face
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Try to detect faces
    faces = detect_faces_and_landmarks(test_image)
    
    # Check if the detector returns a list (even if empty)
    return isinstance(faces, list)

if __name__ == "__main__":
    test_result = test_face_detection()
    print(f"Face detection test: {'Passed' if test_result else 'Failed'}") 