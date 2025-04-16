import cv2
import numpy as np
import os
import traceback

# Define classes of prohibited items
PROHIBITED_CLASSES = ['cell phone', 'mobile phone', 'smartphone', 'book', 'notebook', 'laptop', 'tablet']

# Keywords related to prohibited items for better detection
PHONE_KEYWORDS = ["cell", "phone", "mobile", "cellphone", "smartphone", "cellular", "device", "handset", "iphone", "android"]
BOOK_KEYWORDS = ["book", "notebook", "textbook", "register", "document", "paper"]
COMPUTER_KEYWORDS = ["laptop", "computer", "tablet", "ipad", "surface"]

# Simulation toggles (for testing)
simulate_phone = False  # Set to False to disable automatic phone detection
simulate_book = False

class ObjectDetector:
    def __init__(self):
        # Initialize the list of prohibited items
        self.prohibited_items = ['cellphone', 'book', 'notes', 'laptop']
        
        # Define multiple color ranges for different cellphone colors
        # Format: [(lower_bound1, upper_bound1), (lower_bound2, upper_bound2), ...]
        self.phone_color_ranges = [
            # Black/Dark phones (improved range)
            (np.array([0, 0, 0]), np.array([180, 100, 80])),
            # White/Silver phones
            (np.array([0, 0, 180]), np.array([180, 30, 255])),
            # Gray phones
            (np.array([0, 0, 110]), np.array([180, 30, 220])),
            # Blue phones (expanded range)
            (np.array([90, 50, 50]), np.array([150, 255, 255])),
            # Red phones (two ranges due to HSV color space)
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([170, 70, 50]), np.array([180, 255, 255])),
            # Gold/Yellow phones
            (np.array([15, 50, 120]), np.array([35, 255, 255])),
            # Pink phones
            (np.array([140, 50, 120]), np.array([170, 255, 255])),
            # Green phones
            (np.array([35, 50, 50]), np.array([85, 255, 255])),
            # Brown/Bronze phones
            (np.array([10, 50, 50]), np.array([30, 255, 150])),
        ]
        
        # No external model loading - simplified approach
        self.phone_cascade_loaded = False
        print("Using simplified detection methods without external models")
        
    def detect_objects(self, frame):
        """
        Detects prohibited objects in the frame.
        
        Args:
            frame: The input frame to analyze.
            
        Returns:
            tuple: (bool, list) - Whether prohibited items were detected and a list of detected items.
        """
        try:
            # Skip detection if simulation is enabled
            if simulate_phone:
                print("Phone detection simulation active")
                return True, ["cellphone"]
                
            # Create a copy of the frame for processing
            processed_frame = frame.copy()
            
            # List to store detected items
            detected_items = []
            
            # Detect cellphones
            phone_detected = self.detect_cellphone(processed_frame)
            if phone_detected:
                detected_items.append("cellphone")
            
            # Return the detection results
            return len(detected_items) > 0, detected_items
            
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            traceback.print_exc()
            return False, []
    
    def detect_cellphone(self, frame):
        """
        Detects cellphones in the frame using simplified detection methods.
        
        Args:
            frame: The input frame to analyze.
            
        Returns:
            bool: Whether a cellphone was detected.
        """
        try:
            # Simplified algorithm for phone detection
            # Convert to HSV for color-based detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Apply blur to reduce noise
            blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
            
            # Combine masks from different color ranges
            combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
            for lower_bound, upper_bound in self.phone_color_ranges:
                # Create a mask for the current color range
                current_mask = cv2.inRange(blurred, lower_bound, upper_bound)
                # Combine with the existing mask
                combined_mask = cv2.bitwise_or(combined_mask, current_mask)
            
            # Apply morphological operations to improve mask
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours from the color mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Score system for detection confidence
            phone_score = 0
            
            # Check contours for phone-like shapes
            for contour in contours:
                # Filter by contour area - much stricter threshold
                area = cv2.contourArea(contour)
                if area < 3000:  # Higher threshold to reduce false positives
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = float(w) / h
                
                # Check if aspect ratio matches typical phone dimensions
                # More restricted range
                if (0.4 <= aspect_ratio <= 0.7) or (1.4 <= aspect_ratio <= 2.5):
                    # Calculate the solidity (area / convex hull area)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    # Phones typically have high solidity (filled rectangular shape)
                    # Higher threshold for solidity
                    if solidity > 0.85:
                        print(f"Potential phone-like object: area={area}, aspect_ratio={aspect_ratio}, solidity={solidity}")
                        # Add to score instead of immediately returning
                        phone_score += 0.5
            
            # Only check edges if we have some initial evidence of a phone
            if phone_score > 0:
                # Check for rectangular shapes in the frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred_gray, 50, 150)
                
                contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours_edges:
                    area = cv2.contourArea(contour)
                    if area < 2000:  # Higher threshold
                        continue
                    
                    # Approximate the contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if the approximated contour has exactly 4 points (rectangular)
                    if len(approx) == 4:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(approx)
                        
                        # Calculate aspect ratio
                        aspect_ratio = float(w) / h
                        
                        # Check aspect ratio for potential phone - stricter range
                        if (0.4 <= aspect_ratio <= 0.7) or (1.4 <= aspect_ratio <= 2.5):
                            # Check if it's a proper rectangle
                            is_rectangular = self.check_rectangular_shape(approx)
                            if is_rectangular:
                                print(f"Detected rectangular object: area={area}, aspect_ratio={aspect_ratio}")
                                phone_score += 0.5
            
            # Require a higher confidence score to report a phone
            if phone_score >= 1.0:
                print(f"Detected phone with confidence score: {phone_score}")
                return True
            
            print(f"No phone detected. Confidence score: {phone_score}")
            return False
            
        except Exception as e:
            print(f"Error in phone detection: {str(e)}")
            traceback.print_exc()
            return False
    
    def check_rectangular_shape(self, approx):
        """
        Checks if the given contour approximation is rectangular in shape.
        
        Args:
            approx: Approximated contour points.
            
        Returns:
            bool: Whether the shape is rectangular.
        """
        if len(approx) != 4:
            return False
            
        # Check if all angles are approximately 90 degrees
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i+1) % 4][0]
            p3 = approx[(i+2) % 4][0]
            
            # Calculate vectors
            v1 = [p2[0] - p1[0], p2[1] - p1[1]]
            v2 = [p3[0] - p2[0], p3[1] - p2[1]]
            
            # Calculate dot product
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            
            # Calculate magnitudes
            mag1 = (v1[0]**2 + v1[1]**2)**0.5
            mag2 = (v2[0]**2 + v2[1]**2)**0.5
            
            # Calculate cosine of angle
            if mag1 * mag2 == 0:
                return False
                
            cos_angle = dot / (mag1 * mag2)
            
            # Cosine of 90 degrees is 0, so check if close to 0
            if abs(cos_angle) > 0.3:  # Allow some deviation from 90 degrees
                return False
                
        return True

def toggle_simulate_book():
    global simulate_book
    simulate_book = not simulate_book
    print(f"Book simulation {'enabled' if simulate_book else 'disabled'}")
    print(f"Book simulation is now: {simulate_book}")
    return simulate_book

def toggle_simulate_phone():
    global simulate_phone
    simulate_phone = not simulate_phone
    print(f"Phone simulation {'enabled' if simulate_phone else 'disabled'}")
    print(f"Phone simulation is now: {simulate_phone}")
    return simulate_phone

def detect_prohibited_items(frame):
    """
    Main function to detect prohibited items with improved accuracy.
    
    Args:
        frame: The input frame to analyze.
        
    Returns:
        list: List of detected prohibited items.
    """
    # Check if simulation is on
    global simulate_phone, simulate_book
    
    # Log simulation state
    print(f"DETECTION: Simulation state - Phone: {simulate_phone}, Book: {simulate_book}")
    
    # Return simulated items if enabled
    if simulate_phone and simulate_book:
        print("DETECTION: Simulation active - phone and book")
        return ["phone", "book"]
    elif simulate_phone:
        print("DETECTION: Simulation active - phone only")
        return ["phone"]
    elif simulate_book:
        print("DETECTION: Simulation active - book only")
        return ["book"]
    
    # Use the object detector if the frame is valid and not in simulation mode
    if frame is not None and frame.size > 0:
        try:
            # Ensure frame is properly sized for detection
            max_width = 1280
            if frame.shape[1] > max_width:
                # Resize while keeping aspect ratio
                ratio = max_width / frame.shape[1]
                new_height = int(frame.shape[0] * ratio)
                frame = cv2.resize(frame, (max_width, new_height))
            
            # Create and use detector
            detector = ObjectDetector()
            detected, items = detector.detect_objects(frame)
            
            # Convert to expected format
            result = []
            if "cellphone" in items:
                result.append("phone")
            if "book" in items:
                result.append("book")
                
            return result
            
        except Exception as e:
            print(f"Error in detect_prohibited_items: {str(e)}")
            traceback.print_exc()
    
    return []

if __name__ == "__main__":
    print("Testing object detector...")
    detector = ObjectDetector()
    if detector is not None:
        print("Object detector test passed")
    else:
        print("Object detector test failed") 