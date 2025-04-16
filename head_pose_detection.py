import cv2
import numpy as np
import os
import math
import dlib
import traceback
from pathlib import Path

# Path to facial landmark predictor model
SHAPE_PREDICTOR_PATH = 'shape_predictor_model/shape_predictor_68_face_landmarks.dat'

# Initialize variables globally
face_detector = None
landmark_predictor = None
predictor_loaded = False

# Check if dlib is available
try:
    import dlib
    DLIB_AVAILABLE = True

    # Check if the landmark predictor model exists
    if os.path.exists(SHAPE_PREDICTOR_PATH):
        # Initialize dlib face detector and landmark predictor
        face_detector = dlib.get_frontal_face_detector()
        landmark_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        predictor_loaded = True
        print("Facial landmark predictor loaded successfully for head pose detection")
    else:
        print(f"Warning: Could not find facial landmark predictor model file")
        print(f"Expected at: {SHAPE_PREDICTOR_PATH}")
        # Initialize face detector anyway for the fallback method
        face_detector = dlib.get_frontal_face_detector()
except ImportError:
    print("Warning: dlib not available, head pose detection will be limited")
    DLIB_AVAILABLE = False

def get_landmarks(frame, face_rect):
    """Extract the 68 facial landmarks from a face rectangle"""
    global landmark_predictor
    
    if landmark_predictor is None:
        return None
    
    try:
        # Convert dlib rectangle to format expected by predictor
        if isinstance(face_rect, tuple):
            # Convert tuple (x, y, w, h) to dlib rectangle
            x, y, w, h = face_rect
            rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
        else:
            rect = face_rect
            
        # Get landmarks
        landmarks = landmark_predictor(frame, rect)
        points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append((x, y))
        return points
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return None

def extract_eye(frame, landmarks, eye_points):
    """Extract eye region based on landmarks"""
    try:
        # Get the bounding rectangle of the eye
        eye_region = np.array([(landmarks[i][0], landmarks[i][1]) for i in eye_points], np.int32)
        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])
        
        # Extract the eye region from the frame
        eye = frame[min_y:max_y, min_x:max_x]
        
        # Create a mask for the eye
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [eye_region], 255)
        masked_eye = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Crop the masked eye
        masked_eye = masked_eye[min_y:max_y, min_x:max_x]
        
        return eye, masked_eye, (min_x, min_y, max_x, max_y)
    except Exception as e:
        print(f"Error extracting eye: {e}")
        return None, None, None

def detect_gaze(frame, landmarks):
    """Detect eye gaze direction based on iris position relative to eye corners"""
    if not landmarks or len(landmarks) < 68:
        return "Unknown"
    
    try:
        # Define points for left and right eyes
        left_eye_points = [36, 37, 38, 39, 40, 41]  # Left eye landmarks
        right_eye_points = [42, 43, 44, 45, 46, 47]  # Right eye landmarks
        
        # Extract left and right eyes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, left_eye_masked, left_coords = extract_eye(gray, landmarks, left_eye_points)
        _, right_eye_masked, right_coords = extract_eye(gray, landmarks, right_eye_points)
        
        if left_eye_masked is None or right_eye_masked is None:
            return "Unknown"
        
        # Calculate gaze ratio for each eye
        gaze_ratio_left = calculate_gaze_ratio(left_eye_masked)
        gaze_ratio_right = calculate_gaze_ratio(right_eye_masked)
        
        # Average the gaze ratios
        gaze_ratio = (gaze_ratio_left + gaze_ratio_right) / 2
        
        # Determine gaze direction based on ratio
        if gaze_ratio <= 0.85:
            return "Looking Right"
        elif gaze_ratio >= 1.50:
            return "Looking Left"
        
        # Check for looking up/down based on eye aspect ratio
        left_eye_aspect = calculate_eye_aspect_ratio(landmarks, left_eye_points)
        right_eye_aspect = calculate_eye_aspect_ratio(landmarks, right_eye_points)
        eye_aspect = (left_eye_aspect + right_eye_aspect) / 2
        
        if eye_aspect < 0.2:  # Eyes nearly closed - looking down
            return "Looking Down"
        
        return "Looking Straight"
    except Exception as e:
        print(f"Error in gaze detection: {e}")
        return "Unknown"

def calculate_gaze_ratio(eye):
    """Calculate the ratio of white pixels on left vs right side of eye"""
    try:
        if eye is None or eye.size == 0:
            return 1.0  # Default to center if eye region is empty
            
        height, width = eye.shape
        
        if width <= 1:  # Avoid division by zero
            return 1.0
            
        # Threshold to isolate iris and pupil (dark parts)
        _, threshold_eye = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY)
        
        # Invert to get white on black
        threshold_eye = cv2.bitwise_not(threshold_eye)
        
        # Split eye into left and right parts
        left_side = threshold_eye[0:height, 0:int(width/2)]
        right_side = threshold_eye[0:height, int(width/2):width]
        
        # Count white pixels on each side (add 1 to avoid division by zero)
        left_white = cv2.countNonZero(left_side) + 1
        right_white = cv2.countNonZero(right_side) + 1
        
        # Calculate and return ratio
        return left_white / right_white
    except Exception as e:
        print(f"Error calculating gaze ratio: {e}")
        return 1.0  # Default to center

def calculate_eye_aspect_ratio(landmarks, eye_points):
    """Calculate eye aspect ratio to detect blinks/eye closure"""
    try:
        # Vertical eye landmarks (top, bottom)
        v1 = landmarks[eye_points[1]]
        v2 = landmarks[eye_points[5]]
        
        # Horizontal eye landmarks (left, right)
        h1 = landmarks[eye_points[0]]
        h2 = landmarks[eye_points[3]]
        
        # Calculate distances
        v_dist = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
        h_dist = math.sqrt((h1[0] - h2[0])**2 + (h1[1] - h2[1])**2)
        
        # Calculate ratio
        if h_dist == 0:
            return 0
        return v_dist / h_dist
    except Exception as e:
        print(f"Error calculating eye aspect ratio: {e}")
        return 0.3  # Default value

def simple_head_pose(frame, face_rect):
    """Simple fallback method for head pose detection when landmarks unavailable"""
    # If we have a face but no landmarks, we can at least do a simple check
    # based on the face rectangle
    
    # Convert dlib rectangle to (x,y,w,h) if needed
    if not isinstance(face_rect, tuple):
        x = face_rect.left()
        y = face_rect.top()
        w = face_rect.width()
        h = face_rect.height()
    else:
        x, y, w, h = face_rect
    
    # Compute the face center
    center_x = x + w//2
    center_y = y + h//2
    
    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]
    
    # Check if face is centered in frame
    h_offset = abs(center_x - frame_w//2) / (frame_w//2)
    v_offset = abs(center_y - frame_h//2) / (frame_h//2)
    
    # Simple heuristic for pose
    if h_offset > 0.3:
        return "Looking " + ("Right" if center_x > frame_w//2 else "Left")
    elif v_offset > 0.3:
        return "Looking " + ("Down" if center_y > frame_h//2 else "Up")
    else:
        return "Normal"

def calculate_head_pose(landmarks):
    """Calculate head pose from facial landmarks"""
    if not landmarks or len(landmarks) < 68:
        return "Unknown"
    
    try:
        # Get key facial landmarks
        # Nose tip
        nose_tip = landmarks[30]
        
        # Chin
        chin = landmarks[8]
        
        # Left eye corner
        left_eye = landmarks[36]
        
        # Right eye corner
        right_eye = landmarks[45]
        
        # Mouth corners
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]
        
        # Calculate face center
        face_center_x = (left_eye[0] + right_eye[0]) // 2
        face_center_y = (left_eye[1] + right_eye[1] + chin[1]) // 3
        
        # Calculate vertical and horizontal ratios
        eye_width = abs(right_eye[0] - left_eye[0])
        if eye_width == 0:  # Avoid division by zero
            return "Unknown"
            
        # Nose deviation from center (horizontal)
        nose_deviation_x = (nose_tip[0] - face_center_x) / eye_width
        
        # Mouth deviation from horizontal
        mouth_slope = abs((right_mouth[1] - left_mouth[1]) / max(1, right_mouth[0] - left_mouth[0]))
        
        # Classify pose
        if abs(nose_deviation_x) > 0.5:
            # Head turned significantly left or right
            return "Looking " + ("Right" if nose_deviation_x > 0 else "Left")
        elif mouth_slope > 0.2:
            # Head tilted
            return "Tilted"
        else:
            # Check if looking down or up based on nose position relative to eye line
            eye_line_y = (left_eye[1] + right_eye[1]) // 2
            if (nose_tip[1] - eye_line_y) < 0.3 * abs(chin[1] - eye_line_y):
                return "Looking Up"
            elif (nose_tip[1] - eye_line_y) > 0.7 * abs(chin[1] - eye_line_y):
                return "Looking Down"
            else:
                return "Normal"
    except Exception as e:
        print(f"Error calculating head pose: {e}")
        return "Unknown"

def detect_head_pose(frame, face_rect=None):
    """Detect head pose from frame"""
    global landmark_predictor, face_detector
    
    try:
        # If no face rectangle provided, detect faces
        if face_rect is None:
            if face_detector is None:
                return "Not Available"  # No face detector available
                
            face_rects = face_detector(frame, 1)
            if len(face_rects) == 0:
                return "No Face"
            face_rect = face_rects[0]
        
        # Try landmark-based head pose detection first
        if predictor_loaded and landmark_predictor is not None:
            landmarks = get_landmarks(frame, face_rect)
            if landmarks:
                head_pose = calculate_head_pose(landmarks)
                
                # Check eye gaze separately - more sensitive
                gaze_direction = detect_gaze(frame, landmarks)
                
                # If head pose is normal but gaze is off, report the gaze
                if head_pose == "Normal" and gaze_direction != "Looking Straight":
                    return gaze_direction
                
                return head_pose
        
        # If landmarks failed or predictor not available, use simple method
        return simple_head_pose(frame, face_rect)
    except Exception as e:
        print(f"Error in head pose detection: {e}")
        return "Normal"  # Default to normal instead of error

def draw_head_pose(image, pose):
    """Draw head pose information on an image"""
    if pose == "Normal" or pose == "Looking Straight":
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red
    
    cv2.putText(image, f"Head Pose: {pose}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return image

class HeadPoseDetector:
    # Define facial landmarks indices
    # Left eye indices
    LEFT_EYE_INDICES = list(range(36, 42))
    # Right eye indices
    RIGHT_EYE_INDICES = list(range(42, 48))
    
    def __init__(self):
        try:
            # Path to the shape predictor model file
            self.model_path = os.path.join('shape_predictor_model', 'shape_predictor_68_face_landmarks.dat')
            
            # Check if the model file exists
            if not os.path.isfile(self.model_path):
                print(f"Warning: Facial landmark predictor model file not found at {self.model_path}")
                self.predictor = None
            else:
                # Initialize the dlib face detector and facial landmark predictor
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor(self.model_path)
                print(f"Loaded facial landmark predictor model from {self.model_path}")
        except Exception as e:
            print(f"Warning: Failed to initialize facial landmark predictor. Error: {str(e)}")
            traceback.print_exc()
            self.predictor = None
    
    def detect_head_pose(self, frame):
        """
        Detects the head pose from the input frame and determines if it's normal.
        
        Args:
            frame: Input frame from the webcam.
            
        Returns:
            tuple: (is_normal, angle, message) - 
                  Whether the head pose is normal, the estimated angle, and a message describing the pose.
        """
        try:
            if self.predictor is None:
                return True, 0, "Head pose detection not available (model not loaded)"
            
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale frame
            faces = self.detector(gray, 0)
            
            if len(faces) == 0:
                return False, 0, "No face detected"
            
            # Get the first detected face
            face = faces[0]
            
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            
            # Extract key points for head pose estimation
            nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
            chin = (landmarks.part(8).x, landmarks.part(8).y)
            left_eye_left = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye_right = (landmarks.part(45).x, landmarks.part(45).y)
            left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
            right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
            
            # Calculate head tilt angle based on eye positions
            eye_angle = math.degrees(math.atan2(
                right_eye_right[1] - left_eye_left[1],
                right_eye_right[0] - left_eye_left[0]
            ))
            
            # Calculate distance between eyes
            eye_dist = math.sqrt((right_eye_right[0] - left_eye_left[0])**2 + 
                                 (right_eye_right[1] - left_eye_left[1])**2)
            
            # Calculate distance from nose to chin
            nose_chin_dist = math.sqrt((nose_tip[0] - chin[0])**2 + 
                                       (nose_tip[1] - chin[1])**2)
            
            # Calculate ratio of nose-chin distance to eye distance (proxy for looking up/down)
            vertical_ratio = nose_chin_dist / eye_dist if eye_dist > 0 else 0
            
            # Calculate horizontal head position
            face_center_x = (face.left() + face.right()) // 2
            frame_center_x = frame.shape[1] // 2
            horizontal_deviation = abs(face_center_x - frame_center_x) / frame.shape[1]
            
            # Determine if head pose is normal
            is_normal = (abs(eye_angle) < 15 and  # Head not tilted too much
                        0.5 < vertical_ratio < 1.5 and  # Not looking up or down too much
                        horizontal_deviation < 0.2)  # Not turned too far left or right
            
            # Determine the message based on the pose
            if abs(eye_angle) >= 15:
                message = "Head is tilted"
            elif vertical_ratio <= 0.5:
                message = "Looking down"
            elif vertical_ratio >= 1.5:
                message = "Looking up"
            elif horizontal_deviation >= 0.2:
                if face_center_x < frame_center_x:
                    message = "Looking left"
                else:
                    message = "Looking right"
            else:
                message = "Normal head pose"
            
            return is_normal, eye_angle, message
            
        except Exception as e:
            print(f"Error in head pose detection: {str(e)}")
            traceback.print_exc()
            return True, 0, "Error in head pose detection"
    
    def detect_eye_gaze(self, frame):
        """
        Detects the eye gaze direction from the input frame.
        
        Args:
            frame: Input frame from the webcam.
            
        Returns:
            tuple: (is_looking_straight, direction, confidence, message) - 
                  Whether the person is looking straight ahead, the gaze direction, confidence level, and a message.
        """
        try:
            if self.predictor is None:
                print("Eye gaze detection predictor not loaded")
                # Return a default value instead of failing
                return True, "straight", 0.6, "Looking straight ahead (fallback)"
            
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale frame
            faces = self.detector(gray, 0)
            
            if len(faces) == 0:
                print("No face detected for eye gaze")
                # Return a default value when no face is detected
                return True, "straight", 0.5, "Looking straight ahead (no face)"
            
            # Get the first detected face
            face = faces[0]
            
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            
            # Extract eye regions
            left_eye = self.extract_eye_region(gray, landmarks, self.LEFT_EYE_INDICES)
            right_eye = self.extract_eye_region(gray, landmarks, self.RIGHT_EYE_INDICES)
            
            # Detect gaze for each eye
            left_gaze, left_confidence = self.detect_single_eye_gaze(left_eye)
            right_gaze, right_confidence = self.detect_single_eye_gaze(right_eye)
            
            print(f"Eye gaze detection: Left eye: {left_gaze} ({left_confidence:.2f}), Right eye: {right_gaze} ({right_confidence:.2f})")
            
            # Combine the gaze directions with weighted confidence
            if left_confidence > 0 and right_confidence > 0:
                # If both eyes give valid results, use the one with higher confidence
                if left_confidence > right_confidence:
                    gaze_direction = left_gaze
                    confidence = left_confidence
                else:
                    gaze_direction = right_gaze
                    confidence = right_confidence
            elif left_confidence > 0:
                gaze_direction = left_gaze
                confidence = left_confidence
            elif right_confidence > 0:
                gaze_direction = right_gaze
                confidence = right_confidence
            else:
                # Default to straight if both eyes failed
                gaze_direction = "straight"
                confidence = 0.5
            
            # Determine if the person is looking straight ahead
            is_looking_straight = (gaze_direction == "straight" and confidence > 0.4) or confidence < 0.3
            
            # Create an informative message
            if gaze_direction == "left":
                message = "Looking to the left"
            elif gaze_direction == "right":
                message = "Looking to the right"
            elif gaze_direction == "up":
                message = "Looking up"
            elif gaze_direction == "down":
                message = "Looking down"
            else:
                message = "Looking straight ahead"
            
            return is_looking_straight, gaze_direction, confidence, message
            
        except Exception as e:
            print(f"Error in eye gaze detection: {str(e)}")
            traceback.print_exc()
            # Return a default value instead of failing
            return True, "straight", 0.5, "Looking straight ahead (error handled)"
    
    def extract_eye_region(self, gray, landmarks, eye_indices):
        """
        Extracts the eye region based on facial landmarks.
        
        Args:
            gray: Grayscale image
            landmarks: Facial landmarks
            eye_indices: Indices corresponding to the eye region
            
        Returns:
            numpy.ndarray: Cropped eye region
        """
        # Extract the (x, y) coordinates of the eye landmarks
        eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices]
        
        # Convert eye points to numpy array
        eye_points = np.array(eye_points, dtype=np.int32)
        
        # Get bounding box of eye region
        x, y, w, h = cv2.boundingRect(eye_points)
        
        # Add some margin to the bounding box
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2*margin)
        h = min(gray.shape[0] - y, h + 2*margin)
        
        # Extract the eye region
        eye_region = gray[y:y+h, x:x+w]
        
        # Check if the eye region is valid
        if eye_region.size == 0 or w <= 0 or h <= 0:
            return None
        
        return eye_region
    
    def detect_single_eye_gaze(self, eye_region):
        """
        Detects the gaze direction for a single eye.
        
        Args:
            eye_region: Cropped eye region image
            
        Returns:
            tuple: (direction, confidence) - The detected gaze direction and confidence level
        """
        if eye_region is None or eye_region.size == 0:
            return "unknown", 0.0
        
        try:
            # Make sure the eye region is large enough to process
            if eye_region.shape[0] < 10 or eye_region.shape[1] < 10:
                return "straight", 0.5  # Default to straight for tiny regions
            
            # Apply histogram equalization to enhance contrast
            eye_region = cv2.equalizeHist(eye_region)
            
            # Apply multiple thresholding to catch different pupil brightness levels
            _, thresholded1 = cv2.threshold(eye_region, 40, 255, cv2.THRESH_BINARY_INV)
            _, thresholded2 = cv2.threshold(eye_region, 60, 255, cv2.THRESH_BINARY_INV)
            thresholded = cv2.bitwise_or(thresholded1, thresholded2)
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((3, 3), np.uint8)
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # Try again with a different threshold if no contours found
                _, thresholded = cv2.threshold(eye_region, 30, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    return "straight", 0.5  # Default to straight if still no contours
            
            # Get the largest contour (likely the pupil)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for pupil_contour in contours[:2]:  # Try the two largest contours
                # Get the moments of the pupil contour
                M = cv2.moments(pupil_contour)
                
                if M["m00"] == 0:
                    continue
                
                # Calculate the center of the pupil
                pupil_cx = int(M["m10"] / M["m00"])
                pupil_cy = int(M["m01"] / M["m00"])
                
                # Calculate the center of the eye region
                eye_cx = eye_region.shape[1] // 2
                eye_cy = eye_region.shape[0] // 2
                
                # Calculate the relative position of the pupil within the eye
                rel_x = (pupil_cx - eye_cx) / (eye_region.shape[1] / 2.0)
                rel_y = (pupil_cy - eye_cy) / (eye_region.shape[0] / 2.0)
                
                # Calculate the Euclidean distance from the center
                distance = math.sqrt(rel_x**2 + rel_y**2)
                
                # Calculate a confidence based on the pupil size and distance from center
                pupil_area = cv2.contourArea(pupil_contour)
                max_expected_area = eye_region.shape[0] * eye_region.shape[1] * 0.3  # 30% of eye region
                
                # Normalize the pupil area
                area_ratio = min(1.0, pupil_area / max_expected_area if max_expected_area > 0 else 0)
                
                # Higher confidence if the pupil is larger and further from center
                confidence = area_ratio * min(1.0, distance * 2.0)
                
                # Determine the gaze direction based on the relative position
                if distance < 0.25:  # Increased threshold for "straight"
                    direction = "straight"
                    confidence = max(0.5, confidence)  # Ensure minimum confidence for straight gaze
                else:
                    # Determine the main direction based on the angle
                    angle = math.degrees(math.atan2(rel_y, rel_x))
                    
                    if -45 <= angle < 45:
                        direction = "right"
                    elif 45 <= angle < 135:
                        direction = "down"
                    elif -135 <= angle < -45:
                        direction = "up"
                    else:
                        direction = "left"
                
                # If we got a valid result, return it
                if confidence > 0:
                    return direction, confidence
            
            # If we couldn't get a valid result from any contour
            return "straight", 0.5
        except Exception as e:
            print(f"Error in single eye gaze detection: {str(e)}")
            return "straight", 0.5

# Test function to check if the module works correctly
def test_head_pose_detector():
    detector = HeadPoseDetector()
    print("Head pose detector initialized")
    return detector is not None

if __name__ == "__main__":
    print("Testing head pose detector...")
    if test_head_pose_detector():
        print("Head pose detector test passed")
    else:
        print("Head pose detector test failed") 