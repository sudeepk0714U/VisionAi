import cv2
import speech_recognition as sr
import threading
import asyncio
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from edge_tts import Communicate
from playsound import playsound
import time  # Added for calibration timestamp

model = YOLO("yolov8n.pt")

obstacle_labels = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
}

high_priority_objects = {
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'stairs', 'crosswalk'
}

AUTO_WARNING_OBJECTS = {
    'car', 'truck', 'bus', 'motorcycle', 'bicycle',  # Vehicles
}

SAFETY_DISTANCES = {
    'car': 5.0,
    'truck': 6.0,
    'bus': 6.0,
    'motorcycle': 4.0,
    'bicycle': 3.0,
    'person': 0.4,
    'stairs': 3.0,
    'crosswalk': 4.0,
    'default': 1.5
}

# Camera calibration constants (improved accuracy)
FOCAL_LENGTH = 1000  # pixels - will be calibrated dynamically
KNOWN_OBJECT_WIDTH = 0.5  # meters (average person width)

# Enhanced object size database (more accurate measurements)
OBJECT_SIZES = {
    # People and animals
    'person': 0.45,  # Average shoulder width (more accurate)
    'cat': 0.25,  # Average cat width
    'dog': 0.35,  # Average dog width
    'horse': 0.6,  # Average horse width
    'sheep': 0.4,  # Average sheep width
    'cow': 0.7,  # Average cow width
    'elephant': 1.2,  # Average elephant width
    'bear': 0.8,  # Average bear width
    'zebra': 0.5,  # Average zebra width
    'giraffe': 0.6,  # Average giraffe width

    # Vehicles (more precise measurements)
    'car': 1.85,  # Average car width (standard sedan)
    'truck': 2.55,  # Average truck width
    'bus': 2.55,  # Standard bus width
    'motorcycle': 0.85,  # Average motorcycle width
    'bicycle': 0.7,  # Average bicycle width
    'airplane': 3.5,  # Small aircraft width
    'boat': 2.0,  # Average boat width

    # Furniture and objects
    'chair': 0.55,  # Average chair width
    'couch': 2.2,  # Average couch width
    'bed': 1.4,  # Average bed width
    'dining table': 1.8,  # Average table width
    'tv': 1.2,  # Average TV width
    'laptop': 0.35,  # Average laptop width
    'cell phone': 0.08,  # Average phone width

    # Traffic objects
    'traffic light': 0.3,  # Traffic light width
    'stop sign': 0.6,  # Stop sign width
    'parking meter': 0.2,  # Parking meter width
    'fire hydrant': 0.3,  # Fire hydrant width

    # Sports equipment
    'sports ball': 0.22,  # Soccer ball diameter
    'baseball bat': 0.07,  # Baseball bat diameter
    'tennis racket': 0.35,  # Tennis racket width
    'skateboard': 0.2,  # Skateboard width
}

# Enhanced object size database with confidence levels and size ranges
OBJECT_SIZES_ENHANCED = {
    'person': {
        'width': 0.45,  # meters
        'confidence': 0.95,  # High confidence
        'min_width': 0.35,  # Child
        'max_width': 0.55,  # Large adult
        'height_ratio': 2.5  # Height is typically 2.5x width
    },
    'car': {
        'width': 1.85,
        'confidence': 0.90,
        'min_width': 1.65,  # Small car
        'max_width': 2.05,  # Large car
        'height_ratio': 1.4
    },
    'truck': {
        'width': 2.55,
        'confidence': 0.88,
        'min_width': 2.35,
        'max_width': 2.75,
        'height_ratio': 1.8
    },
    'bicycle': {
        'width': 0.7,
        'confidence': 0.92,
        'min_width': 0.6,
        'max_width': 0.8,
        'height_ratio': 1.6
    },
    'motorcycle': {
        'width': 0.85,
        'confidence': 0.89,
        'min_width': 0.75,
        'max_width': 0.95,
        'height_ratio': 1.3
    }
}

# Default fallback for objects not in enhanced database
DEFAULT_OBJECT_INFO = {
    'width': 0.5,
    'confidence': 0.7,
    'min_width': 0.3,
    'max_width': 0.8,
    'height_ratio': 1.5
}


class SafetyNavigationSystem:
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        self.danger_zone_radius = 100  # pixels
        self.last_warning_time = 0
        self.warning_cooldown = 2.0  # seconds
        self.debug_mode = False  # Toggle debug mode
        self.last_spoken_alerts = set()  # Track what alerts were already spoken
        self.alert_cooldown = 3.0  # Seconds between repeating the same alert

        # Enhanced distance calibration data
        self.calibration_points = []
        self.focal_length_calibrated = False
        self.actual_focal_length = FOCAL_LENGTH

        # Multi-frame tracking for improved accuracy
        self.object_tracking_history = {}  # Track objects across frames
        self.frame_count = 0
        self.motion_vectors = {}  # Store motion vectors for parallax

        # Camera movement detection
        self.previous_frame_objects = []
        self.camera_moving = False
        self.movement_threshold = 5.0  # pixels

        # Distance estimation confidence tracking
        self.distance_confidence_history = {}
        self.min_confidence_threshold = 0.3

        # Calibration quality metrics
        self.calibration_quality = 0.0
        self.calibration_samples = 0
        self.required_calibration_samples = 5

    def toggle_debug(self):
        """Toggle debug mode on/off"""
        self.debug_mode = not self.debug_mode
        print(f"[DEBUG] Debug mode {'ON' if self.debug_mode else 'OFF'}")

    def get_debug_info(self):
        """Get current debug information"""
        return f"Debug Mode: {'ON' if self.debug_mode else 'OFF'}"

    def should_speak_alert(self, alert_text):
        """Check if an alert should be spoken (avoid spam)"""
        import time
        current_time = time.time()

        # Create a simplified key for the alert
        alert_key = alert_text[:50]  # First 50 characters

        # Check if we've spoken this alert recently
        if alert_key in self.last_spoken_alerts:
            return False

        # Add to spoken alerts and set cooldown
        self.last_spoken_alerts.add(alert_key)

        # Clean up old alerts after cooldown
        if current_time - self.last_warning_time > self.alert_cooldown:
            self.last_spoken_alerts.clear()
            self.last_warning_time = current_time

        return True

    def get_object_direction(self, relative_x):
        """Get simple direction of the object (left, right, or front)."""
        if relative_x > 0.3:
            return "to your right"
        elif relative_x < -0.3:
            return "to your left"
        else:
            return "in front of you"

    def calibrate_distance(self, known_distance_meters, object_width_pixels, object_type):
        """Enhanced calibration with multiple points and quality assessment"""
        if object_type in OBJECT_SIZES:
            known_width = OBJECT_SIZES[object_type]
            # focal_length = (pixel_width * distance) / real_width
            calculated_focal = (object_width_pixels * known_distance_meters) / known_width

            # Store calibration point
            calibration_point = {
                'focal_length': calculated_focal,
                'distance': known_distance_meters,
                'object_type': object_type,
                'confidence': 0.8,  # Base confidence
                'timestamp': time.time()
            }

            self.calibration_points.append(calibration_point)
            self.calibration_samples += 1

            # Calculate focal length using statistical methods
            if self.calibration_samples >= 3:
                # Use median for robustness against outliers
                focal_lengths = [cp['focal_length'] for cp in self.calibration_points[-5:]]  # Last 5 samples
                median_focal = np.median(focal_lengths)

                # Calculate standard deviation to assess calibration quality
                focal_std = np.std(focal_lengths)
                focal_mean = np.mean(focal_lengths)

                # Quality metric: lower std/mean ratio = better quality
                if focal_mean > 0:
                    self.calibration_quality = 1.0 / (1.0 + (focal_std / focal_mean))
                else:
                    self.calibration_quality = 0.0

                # Use weighted average based on quality
                if self.calibration_quality > 0.7:  # Good calibration
                    self.actual_focal_length = median_focal
                    self.focal_length_calibrated = True
                    print(
                        f"[CALIBRATION] High quality focal length: {median_focal:.1f}px (quality: {self.calibration_quality:.2f})")
                else:
                    # Use simple average for lower quality calibrations
                    self.actual_focal_length = focal_mean
                    self.focal_length_calibrated = True
                    print(
                        f"[CALIBRATION] Standard focal length: {focal_mean:.1f}px (quality: {self.calibration_quality:.2f})")

                if self.debug_mode:
                    print(
                        f"[DEBUG] Calibration samples: {self.calibration_samples}, Std: {focal_std:.1f}, Quality: {self.calibration_quality:.2f}")
            else:
                # Use single point calibration initially
                self.actual_focal_length = calculated_focal
                self.focal_length_calibrated = True
                print(
                    f"[CALIBRATION] Initial focal length: {calculated_focal:.1f}px (samples: {self.calibration_samples})")

            return True
        return False

    def estimate_distance_advanced(self, object_width_pixels, object_type, confidence=0.5, object_id=None):
        """Advanced distance estimation with multiple methods and confidence weighting"""
        if object_width_pixels == 0:
            return float('inf')

        # Get enhanced object information
        if object_type in OBJECT_SIZES_ENHANCED:
            obj_info = OBJECT_SIZES_ENHANCED[object_type]
            known_width = obj_info['width']
            size_confidence = obj_info['confidence']
            min_width = obj_info['min_width']
            max_width = obj_info['max_width']
        else:
            obj_info = DEFAULT_OBJECT_INFO
            known_width = OBJECT_SIZES.get(object_type, KNOWN_OBJECT_WIDTH)
            size_confidence = obj_info['confidence']
            min_width = obj_info['min_width']
            max_width = obj_info['max_width']

        # Method 1: Standard focal length calculation with size range validation
        distance1 = (known_width * self.actual_focal_length) / object_width_pixels

        # Validate distance using size range
        min_distance = (min_width * self.actual_focal_length) / object_width_pixels
        max_distance = (max_width * self.actual_focal_length) / object_width_pixels

        # If distance is outside reasonable range, adjust confidence
        if distance1 < min_distance or distance1 > max_distance:
            confidence *= 0.7  # Reduce confidence for out-of-range estimates

        # Method 2: Multi-frame averaging for stability
        distance2 = float('inf')
        if object_id and object_id in self.object_tracking_history:
            tracking_info = self.get_object_tracking_info(object_id)
            if tracking_info and tracking_info['frames_tracked'] >= 3:
                # Use average of recent distance estimates
                recent_distances = []
                for det in self.object_tracking_history[object_id][-3:]:
                    if det['width'] > 0:
                        recent_distances.append((known_width * self.actual_focal_length) / det['width'])

                if recent_distances:
                    distance2 = np.mean(recent_distances)
                    # Weight by size stability
                    stability_factor = tracking_info['size_stability']
                    if stability_factor > 0.8:  # High stability
                        confidence *= 1.2
                    elif stability_factor < 0.5:  # Low stability
                        confidence *= 0.8

        # Method 3: Motion parallax depth estimation
        distance3 = float('inf')
        if object_id and object_id in self.motion_vectors and self.camera_moving:
            motion_info = self.motion_vectors[object_id]
            if motion_info['magnitude'] > 2.0:  # Significant motion
                # Simple motion parallax: closer objects move more
                # This is a simplified implementation
                motion_factor = motion_info['magnitude'] / 10.0  # Normalize motion
                # Estimate distance based on motion (closer = more motion)
                distance3 = max(0.5, 5.0 / (1.0 + motion_factor))

                if self.debug_mode:
                    print(f"[DEBUG] Motion parallax distance: {distance3:.2f}m (motion: {motion_factor:.2f})")

        # Method 4: Relative size comparison with other objects
        distance4 = float('inf')
        if hasattr(self, 'last_detected_objects') and len(self.last_detected_objects) > 1:
            for obj in self.last_detected_objects:
                if obj['label'] == object_type and obj['distance'] > 0 and obj['distance'] < 20:
                    # Use relative size ratio for distance estimation
                    size_ratio = object_width_pixels / obj.get('pixel_width', object_width_pixels)
                    if size_ratio > 0.1 and size_ratio < 10:  # Reasonable ratio
                        distance4 = obj['distance'] * size_ratio
                        break

        # Method 5: Confidence-based focal length adjustment
        distance5 = float('inf')
        if self.calibration_quality > 0.8:  # High quality calibration
            # Use calibrated focal length
            distance5 = (known_width * self.actual_focal_length) / object_width_pixels
        else:
            # Use adaptive focal length based on object type
            adaptive_focal = self.actual_focal_length * (0.9 + 0.2 * confidence)
            distance5 = (known_width * adaptive_focal) / object_width_pixels

        # Combine all available distance estimates with weighted averaging
        distances = []
        weights = []

        # Primary method (focal length)
        if distance1 != float('inf'):
            distances.append(distance1)
            weights.append(0.4)  # Primary weight

        # Multi-frame averaging
        if distance2 != float('inf'):
            distances.append(distance2)
            weights.append(0.25)

        # Motion parallax
        if distance3 != float('inf'):
            distances.append(distance3)
            weights.append(0.15)

        # Relative size comparison
        if distance4 != float('inf'):
            distances.append(distance4)
            weights.append(0.15)

        # Confidence-adjusted focal length
        if distance5 != float('inf'):
            distances.append(distance5)
            weights.append(0.05)

        # Calculate weighted average
        if len(distances) > 0:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            weighted_distance = sum(d * w for d, w in zip(distances, weights))

            # Apply confidence-based adjustment
            confidence_factor = 0.8 + (confidence * 0.4)  # 0.8 to 1.2 range
            final_distance = weighted_distance * confidence_factor

            # Apply size confidence adjustment
            final_distance *= (0.9 + size_confidence * 0.2)  # 0.9 to 1.1 range

            # Apply reasonable bounds
            final_distance = max(0.1, min(final_distance, 50.0))  # 0.1m to 50m

            if self.debug_mode and len(distances) > 1:
                print(
                    f"[DEBUG] Distance methods: {len(distances)}, Final: {final_distance:.2f}m, Confidence: {confidence:.2f}")

            return final_distance

        # Fallback to primary method
        return distance1

    def estimate_distance(self, object_width_pixels, object_type, confidence=0.5):
        """Main distance estimation method - now uses advanced calculation"""
        return self.estimate_distance_advanced(object_width_pixels, object_type, confidence)

    def calculate_object_position(self, x1, y1, x2, y2):
        """Calculate object's position relative to camera center"""
        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2

        # Calculate relative position (-1 to 1, where 0 is center)
        relative_x = (object_center_x - self.center_x) / self.center_x
        relative_y = (object_center_y - self.center_y) / self.center_y

        # Calculate distance from center
        distance_from_center = np.sqrt(relative_x ** 2 + relative_y ** 2)

        return relative_x, relative_y, distance_from_center

    def is_in_danger_zone(self, x1, y1, x2, y2, distance):
        """Check if object is in immediate danger zone"""
        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2

        # Check if object is near center of frame (user's path)
        center_distance = np.sqrt((object_center_x - self.center_x) ** 2 +
                                  (object_center_y - self.center_y) ** 2)

        # Object is dangerous if it's close AND in the center path
        return distance < 2.0 and center_distance < self.danger_zone_radius

    def get_direction_instruction(self, relative_x, relative_y):
        """Get directional instruction for avoiding obstacles"""
        if abs(relative_x) < 0.3:  # Object is roughly in front
            if relative_y > 0.3:  # Object is below (closer)
                return "Obstacle directly ahead, stop immediately"
            else:  # Object is above (further)
                return "Obstacle ahead, proceed with caution"
        elif relative_x > 0.3:  # Object is to the right
            return "Obstacle to your right, move left"
        else:  # Object is to the left
            return "Obstacle to your left, move right"

    # def detect_stairs(self, frame):
    #     """Detect stairs using edge detection and line analysis"""
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     # Apply Gaussian blur to reduce noise
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     edges = cv2.Canny(blurred, 50, 150)
    #
    #     # Detect horizontal lines (potential stair edges)
    #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
    #                             minLineLength=120, maxLineGap=15)
    #
    #     if lines is None:
    #         if self.debug_mode:
    #             print("[DEBUG] No lines detected in stair detection")
    #         return False

        stair_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check if line is roughly horizontal (more strict criteria)
            if abs(y2 - y1) < 15 and abs(x2 - x1) > 80:
                horizontal_lines.append(line[0])

                # Check if this line could be part of a stair pattern
                # Look for lines that are roughly parallel and evenly spaced
                if len(horizontal_lines) > 1:
                    # Check spacing between lines
                    for prev_line in horizontal_lines[:-1]:
                        prev_y = (prev_line[1] + prev_line[3]) / 2
                        curr_y = (y1 + y2) / 2
                        spacing = abs(curr_y - prev_y)

                        # Stairs typically have consistent spacing between 20-80 pixels
                        if 20 <= spacing <= 80:
                            stair_lines.append(line[0])
                            if self.debug_mode:
                                print(f"[DEBUG] Stair line detected: spacing={spacing:.1f}px")
                            break

        # Require at least 3 well-spaced horizontal lines for stair detection
        # Also check if lines are distributed across the frame height
        if len(stair_lines) >= 3:
            # Get the y-coordinates of detected lines
            y_coords = []
            for line in stair_lines:
                y_coords.append((line[1] + line[3]) / 2)

            y_coords.sort()

            # Check if lines are distributed across the frame (not clustered)
            total_height = self.frame_height
            min_spacing = total_height / 6  # Lines should be spread across at least 1/6 of frame

            if y_coords[-1] - y_coords[0] >= min_spacing:
                if self.debug_mode:
                    print(
                        f"[DEBUG] STAIRS DETECTED: {len(stair_lines)} lines, height spread={y_coords[-1] - y_coords[0]:.1f}px")
                return True

        if self.debug_mode and len(horizontal_lines) > 0:
            print(f"[DEBUG] Horizontal lines found: {len(horizontal_lines)}, but not enough for stairs")

        return False

    def detect_crosswalk(self, frame):
        """Detect crosswalk using pattern recognition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Look for alternating black and white stripes
        # This is a simplified detection - could be enhanced with ML
        edges = cv2.Canny(blurred, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=100, maxLineGap=8)

        if lines is None:
            if self.debug_mode:
                print("[DEBUG] No lines detected in crosswalk detection")
            return False

        horizontal_lines = 0
        valid_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # More strict criteria for horizontal lines
            if abs(y2 - y1) < 12 and abs(x2 - x1) > 80:
                # Check if line is in the lower part of the frame (where crosswalks typically are)
                line_y = (y1 + y2) / 2
                if line_y > self.frame_height * 0.4:  # Only in lower 60% of frame
                    horizontal_lines += 1
                    valid_lines.append(line[0])

        if self.debug_mode:
            print(f"[DEBUG] Crosswalk detection: {horizontal_lines} horizontal lines found")

        # Require at least 4 horizontal lines for crosswalk detection
        # Crosswalks typically have multiple parallel stripes
        if horizontal_lines >= 4:
            # Check if lines are roughly parallel and evenly spaced
            if len(valid_lines) >= 4:
                y_coords = []
                for line in valid_lines:
                    y_coords.append((line[1] + line[3]) / 2)

                y_coords.sort()

                # Check spacing consistency (crosswalk stripes should be roughly evenly spaced)
                spacings = []
                for i in range(1, len(y_coords)):
                    spacings.append(y_coords[i] - y_coords[i - 1])

                if len(spacings) >= 2:
                    avg_spacing = sum(spacings) / len(spacings)
                    # Check if spacings are consistent (within 30% of average)
                    consistent_spacing = all(abs(spacing - avg_spacing) < avg_spacing * 0.3 for spacing in spacings)

                    if consistent_spacing and 15 <= avg_spacing <= 60:
                        if self.debug_mode:
                            print(
                                f"[DEBUG] CROSSWALK DETECTED: {horizontal_lines} lines, avg spacing={avg_spacing:.1f}px")
                        return True

        return False

    def generate_safety_alert(self, object_type, distance, relative_x, relative_y, is_dangerous):
        """Generate appropriate safety alert message"""
        if is_dangerous:
            direction = self.get_direction_instruction(relative_x, relative_y)
            return f"WARNING! {object_type} {distance:.1f} meters away. {direction}"
        elif distance < SAFETY_DISTANCES.get(object_type, SAFETY_DISTANCES['default']):
            direction = self.get_direction_instruction(relative_x, relative_y)
            return f"Caution: {object_type} {distance:.1f} meters ahead. {direction}"
        else:
            return f"{object_type} detected {distance:.1f} meters away"

    def suggest_path(self, detected_objects):
        """Suggest safe walking path based on detected obstacles"""
        if not detected_objects:
            return "Path ahead is clear, you can proceed safely"

        # Analyze obstacle positions
        left_obstacles = []
        right_obstacles = []
        center_obstacles = []

        for obj in detected_objects:
            if obj['relative_x'] < -0.3:
                left_obstacles.append(obj)
            elif obj['relative_x'] > 0.3:
                right_obstacles.append(obj)
            else:
                center_obstacles.append(obj)

        # Suggest best path
        if len(center_obstacles) == 0:
            return "Path ahead is clear, proceed straight"
        elif len(left_obstacles) < len(right_obstacles):
            return "Obstacles ahead, consider moving to your left"
        else:
            return "Obstacles ahead, consider moving to your right"

    def track_object_across_frames(self, object_id, bbox, label, confidence):
        """Track object across frames for improved distance estimation"""
        current_time = time.time()

        if object_id not in self.object_tracking_history:
            self.object_tracking_history[object_id] = []

        # Store current detection
        detection = {
            'bbox': bbox,
            'label': label,
            'confidence': confidence,
            'timestamp': current_time,
            'center_x': (bbox[0] + bbox[2]) / 2,
            'center_y': (bbox[1] + bbox[3]) / 2,
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1]
        }

        self.object_tracking_history[object_id].append(detection)

        # Keep only recent history (last 10 frames)
        if len(self.object_tracking_history[object_id]) > 10:
            self.object_tracking_history[object_id] = self.object_tracking_history[object_id][-10:]

        # Calculate motion vector if we have enough history
        if len(self.object_tracking_history[object_id]) >= 2:
            prev_detection = self.object_tracking_history[object_id][-2]
            motion_x = detection['center_x'] - prev_detection['center_x']
            motion_y = detection['center_y'] - prev_detection['center_y']

            self.motion_vectors[object_id] = {
                'motion_x': motion_x,
                'motion_y': motion_y,
                'magnitude': np.sqrt(motion_x ** 2 + motion_y ** 2),
                'timestamp': current_time
            }

            # Detect if camera is moving (objects moving in opposite directions)
            if abs(motion_x) > self.movement_threshold or abs(motion_y) > self.movement_threshold:
                self.camera_moving = True
            else:
                self.camera_moving = False

        return len(self.object_tracking_history[object_id])

    def get_object_tracking_info(self, object_id):
        """Get tracking information for an object"""
        if object_id in self.object_tracking_history:
            history = self.object_tracking_history[object_id]
            if len(history) >= 2:
                return {
                    'frames_tracked': len(history),
                    'motion_vector': self.motion_vectors.get(object_id, {}),
                    'size_stability': self.calculate_size_stability(object_id),
                    'confidence_trend': self.calculate_confidence_trend(object_id)
                }
        return None

    def calculate_size_stability(self, object_id):
        """Calculate how stable the object size is across frames"""
        if object_id in self.object_tracking_history:
            history = self.object_tracking_history[object_id]
            if len(history) >= 3:
                widths = [det['width'] for det in history[-3:]]
                width_std = np.std(widths)
                width_mean = np.mean(widths)
                if width_mean > 0:
                    return 1.0 / (1.0 + (width_std / width_mean))
        return 0.0

    def calculate_confidence_trend(self, object_id):
        """Calculate confidence trend across frames"""
        if object_id in self.object_tracking_history:
            history = self.object_tracking_history[object_id]
            if len(history) >= 3:
                confidences = [det['confidence'] for det in history[-3:]]
                # Positive trend if confidence is increasing
                if len(confidences) >= 2:
                    return confidences[-1] - confidences[0]
        return 0.0


# Initialize safety navigation system
safety_system = SafetyNavigationSystem()

# Shared variable for voice command
last_command = ""
lock = threading.Lock()
current_speech_rate = "+20%"  # Default speech rate
voice_thread_running = False  # Track if voice thread is running
last_safety_command_time = 0  # Track when safety command was last processed


# Speech recognition thread
def listen_thread():
    global last_command
    recognizer = sr.Recognizer()

    # Adjust microphone settings for better recognition
    recognizer.energy_threshold = 4000  # Increase energy threshold
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8  # Shorter pause threshold

    with sr.Microphone() as source:
        print("[VOICE] Microphone initialized and listening...")
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while True:
            try:
                print("[VOICE] Listening for command...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

                print("[VOICE] Processing audio...")
                command = recognizer.recognize_google(audio).lower()
                print(f"[VOICE] Heard: '{command}'")

                with lock:
                    last_command = command
                    print(f"[VOICE] Command stored: '{last_command}'")

            except sr.WaitTimeoutError:
                print("[VOICE] No speech detected within timeout")
                continue
            except sr.UnknownValueError:
                print("[VOICE] Speech was unintelligible")
                continue
            except sr.RequestError as e:
                print(f"[VOICE] Could not request results from Google Speech Recognition service: {e}")
                continue
            except Exception as e:
                print(f"[VOICE] Error in speech recognition: {e}")
                continue


# TTS function using edge-tts
async def speak_async(text, rate="+20%"):
    try:
        communicate = Communicate(text=text, voice="en-IN-NeerjaNeural", rate=rate)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            await communicate.save(tmp.name)
            playsound(tmp.name)
            os.remove(tmp.name)
    except Exception as e:
        print("[TTS ERROR]", e)


def speak(text, rate="+20%"):
    asyncio.run(speak_async(text, rate))


# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not accessible")
    exit(1)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Smart AI Assistant with Safety Navigation Active")
print("[INFO] Press 'q' to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
    detected_objects = []
    safety_alerts = []

    # Detect stairs and crosswalks
    # stairs_detected = safety_system.detect_stairs(frame)
    stairs_detected = False

    crosswalk_detected = safety_system.detect_crosswalk(frame)

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        label = model.names[int(cls)]

        if label in obstacle_labels:
            # Calculate object properties
            object_width = x2 - x1
            distance = safety_system.estimate_distance(object_width, label)
            relative_x, relative_y, center_distance = safety_system.calculate_object_position(x1, y1, x2, y2)
            is_dangerous = safety_system.is_in_danger_zone(x1, y1, x2, y2, distance)

            # Store object info
            detected_objects.append({
                'label': label,
                'distance': distance,
                'relative_x': relative_x,
                'relative_y': relative_y,
                'is_dangerous': is_dangerous
            })

            # Generate safety alert
            alert = safety_system.generate_safety_alert(label, distance, relative_x, relative_y, is_dangerous)

            # Check if the alert should be spoken
            if safety_system.should_speak_alert(alert):
                safety_alerts.append(alert)
                # Only speak critical warnings automatically for vehicles and persons
                if is_dangerous and label in AUTO_WARNING_OBJECTS:
                    # **NEW**: Get the direction and add it to the spoken alert
                    direction = safety_system.get_object_direction(relative_x)
                    speak(f"WARNING! {label} dangerously close, {direction}!", current_speech_rate)

            # Visual indicators
            color = (0, 0, 255) if is_dangerous else (0, 255, 0)  # Red for dangerous, green for safe
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {distance:.1f}m", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw distance line
            cv2.line(frame, (int(x1 + object_width / 2), int(y2)),
                     (safety_system.center_x, safety_system.frame_height), (255, 255, 0), 1)

    # Draw safety zone
    cv2.circle(frame, (safety_system.center_x, safety_system.frame_height),
               safety_system.danger_zone_radius, (0, 255, 255), 2)

    # Special detection indicators
    if stairs_detected:
        cv2.putText(frame, "STAIRS DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        stairs_alert = "Stairs detected ahead, be careful with elevation change"
        safety_alerts.append(stairs_alert)
        # Re-enabled automatic stairs warning
        if safety_system.should_speak_alert(stairs_alert):
            speak(stairs_alert, current_speech_rate)

    if crosswalk_detected:
        cv2.putText(frame, "CROSSWALK", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        crosswalk_alert = "Crosswalk detected, check for traffic signals"
        safety_alerts.append(crosswalk_alert)
        # Automatic crosswalk warning remains OFF

    # Listen for voice command
    if not voice_thread_running:
        voice_thread = threading.Thread(target=listen_thread, daemon=True)
        voice_thread.start()
        voice_thread_running = True
        print("[VOICE] Voice recognition thread started")

    # Process last command (The rest of your command processing logic remains unchanged)
    with lock:
        # Debug: Print the current command being processed
        if last_command:
            print(f"[DEBUG] Processing command: '{last_command}'")

        if "what" in last_command and "front" in last_command:
            print("[DEBUG] 'what front' command detected")
            if detected_objects:
                # Enhanced response with directional information
                message = "I see: " + ", ".join(
                    [
                        f"a {obj['label']} {safety_system.get_object_direction(obj['relative_x'])} about {obj['distance']:.1f} meters away"
                        for obj in detected_objects])

                if any(obj['is_dangerous'] for obj in detected_objects):
                    message += ". WARNING: Some objects are dangerously close!"
            else:
                message = "Path ahead is clear, no obstacles detected."

            # Add path suggestion
            path_suggestion = safety_system.suggest_path(detected_objects)
            message += f" {path_suggestion}"

            print("[RESPONSE]", message)
            speak(message, current_speech_rate)
            last_command = ""

        elif "safety" in last_command or "danger" in last_command:
            # Prevent safety command from being processed too frequently (within 2 seconds)
            current_time = time.time()
            if current_time - last_safety_command_time > 2.0:
                print("[DEBUG] 'safety/danger' command detected")
                if safety_alerts:
                    message = "Safety alerts: " + ". ".join(safety_alerts[:1])  # Limit to 1 alert
                    print("[SAFETY]", message)
                    speak(message, current_speech_rate)
                else:
                    message = "No immediate safety concerns detected."
                    print("[SAFETY]", message)
                    speak(message, current_speech_rate)
                last_safety_command_time = current_time
            else:
                print("[DEBUG] Safety command ignored - too frequent")
            last_command = ""

        elif "path" in last_command or "route" in last_command:
            print("[DEBUG] 'path/route' command detected")
            path_suggestion = safety_system.suggest_path(detected_objects)
            print("[PATH]", path_suggestion)
            speak(path_suggestion, current_speech_rate)
            last_command = ""

        elif "debug" in last_command or "toggle debug" in last_command:
            print("[DEBUG] 'debug' command detected")
            safety_system.toggle_debug()
            debug_status = safety_system.get_debug_info()
            print("[DEBUG]", debug_status)
            speak(f"Debug mode {'enabled' if safety_system.debug_mode else 'disabled'}", current_speech_rate)
            last_command = ""

        elif "fast" in last_command and "speech" in last_command:
            print("[DEBUG] 'fast speech' command detected")
            current_speech_rate = "+50%"
            print("[SPEECH] Speed set to fast (+50%)")
            speak("Speech speed set to fast", current_speech_rate)
            last_command = ""

        elif "normal" in last_command and "speech" in last_command:
            print("[DEBUG] 'normal speech' command detected")
            current_speech_rate = "+20%"
            print("[SPEECH] Speed set to normal (+20%)")
            speak("Speech speed set to normal", current_speech_rate)
            last_command = ""

        elif "slow" in last_command and "speech" in last_command:
            print("[DEBUG] 'slow speech' command detected")
            current_speech_rate = "-20%"
            print("[SPEECH] Speed set to slow (-20%)")
            speak("Speech speed set to slow", current_speech_rate)
            last_command = ""

        elif "speed" in last_command and "up" in last_command:
            print("[DEBUG] 'speed up' command detected")
            # Parse current rate and increase it
            try:
                if current_speech_rate.startswith("+"):
                    current_rate = int(current_speech_rate[1:-1])
                    new_rate = min(current_rate + 10, 100)  # Max 100%
                    current_speech_rate = f"+{new_rate}%"
                else:
                    current_rate = int(current_speech_rate[1:-1])
                    new_rate = current_rate + 10
                    current_speech_rate = f"+{new_rate}%"
                print(f"[SPEECH] Speed increased to {current_speech_rate}")
                speak(f"Speech speed increased to {current_speech_rate}", current_speech_rate)
            except:
                current_speech_rate = "+30%"
                print("[SPEECH] Speed reset to +30%")
                speak("Speech speed reset to 30 percent", current_speech_rate)
            last_command = ""

        elif "speed" in last_command and "down" in last_command:
            print("[DEBUG] 'speed down' command detected")
            # Parse current rate and decrease it
            try:
                if current_speech_rate.startswith("+"):
                    current_rate = int(current_speech_rate[1:-1])
                    new_rate = max(current_rate - 10, 10)  # Min 10%
                    current_speech_rate = f"+{new_rate}%"
                else:
                    current_rate = int(current_speech_rate[1:-1])
                    new_rate = max(current_rate - 10, -50)  # Min -50%
                    current_speech_rate = f"{new_rate:+d}%"
                print(f"[SPEECH] Speed decreased to {current_speech_rate}")
                speak(f"Speech speed decreased to {current_speech_rate}", current_speech_rate)
            except:
                current_speech_rate = "+10%"
                print("[SPEECH] Speed reset to +10%")
                speak("Speech speed reset to 10 percent", current_speech_rate)
            last_command = ""

        elif "auto" in last_command and "announce" in last_command:
            print("[DEBUG] 'auto announce' command detected")
            if "on" in last_command or "enable" in last_command:
                safety_system.alert_cooldown = 3.0  # Enable automatic critical warnings
                print("[AUTO] Automatic critical warnings enabled")
                speak("Automatic critical warnings enabled", current_speech_rate)
            elif "off" in last_command or "disable" in last_command:
                safety_system.alert_cooldown = 999999  # Disable automatic critical warnings
                print("[AUTO] Automatic critical warnings disabled")
                speak("Automatic critical warnings disabled", current_speech_rate)
            else:
                # Toggle automatic critical warnings
                if safety_system.alert_cooldown > 1000:
                    safety_system.alert_cooldown = 3.0
                    print("[AUTO] Automatic critical warnings enabled")
                    speak("Automatic critical warnings enabled", current_speech_rate)
                else:
                    safety_system.alert_cooldown = 999999
                    print("[AUTO] Automatic critical warnings disabled")
                    speak("Automatic critical warnings disabled", current_speech_rate)
            last_command = ""

        # If no command was processed, show available commands
        elif last_command:
            print(f"[DEBUG] Unrecognized command: '{last_command}'")
            print(
                "[DEBUG] Available commands: 'what front', 'safety', 'path', 'debug', 'fast speech', 'normal speech', 'slow speech', 'speed up', 'speed down', 'auto announce'")
            # speak("Command not recognized. Try saying 'what front' or 'safety'", current_speech_rate)  # Removed to prevent speaking unrecognized commands
            last_command = ""

        # Test command to verify voice recognition
        elif "test" in last_command or "hello" in last_command:
            print("[DEBUG] 'test/hello' command detected")
            message = "Voice recognition is working! I can hear you clearly."
            print("[TEST]", message)
            speak(message, current_speech_rate)
            last_command = ""

    # Display safety information on screen
    y_offset = 110
    for i, alert in enumerate(safety_alerts[:3]):  # Show top 3 alerts
        cv2.putText(frame, alert[:50] + "...", (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display current speech speed
    speed_text = f"Speech Speed: {current_speech_rate}"
    cv2.putText(frame, speed_text, (10, y_offset + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Display automatic announcements status
    auto_status = "Critical Warnings: AUTO" if safety_system.alert_cooldown < 1000 else "Critical Warnings: OFF"
    auto_color = (0, 255, 0) if safety_system.alert_cooldown < 1000 else (0, 0, 255)
    cv2.putText(frame, auto_status, (10, y_offset + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, auto_color, 2)

    # Display voice command status
    voice_status = f"Voice: {last_command[:30]}" if last_command else "Voice: Listening..."
    voice_color = (0, 255, 0) if last_command else (255, 255, 0)
    cv2.putText(frame, voice_status, (10, y_offset + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)

    # Display available commands hint
    commands_hint = "Say: 'test', 'what front', 'safety', 'debug'"
    cv2.putText(frame, commands_hint, (10, y_offset + 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Display
    cv2.imshow("Smart AI Assistant - Safety Navigation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()