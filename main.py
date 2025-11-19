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
import time
from dotenv import load_dotenv
import openai
from openai import OpenAI
import base64
from io import BytesIO
import face_recognition
import pickle
from pathlib import Path
import json
from datetime import datetime
load_dotenv()
import google.generativeai as genai
from PIL import Image

try:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to your API key.")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("[INFO] OpenAI client initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize OpenAI API: {e}")
    openai_client = None

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
    'car', 'truck', 'bus', 'motorcycle', 'bicycle',
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

FOCAL_LENGTH = 1000
KNOWN_OBJECT_WIDTH = 0.5

OBJECT_SIZES = {
    'person': 0.45,
    'cat': 0.25,
    'dog': 0.35,
    'horse': 0.6,
    'sheep': 0.4,
    'cow': 0.7,
    'elephant': 1.2,
    'bear': 0.8,
    'zebra': 0.5,
    'giraffe': 0.6,
    'car': 1.85,
    'truck': 2.55,
    'bus': 2.55,
    'motorcycle': 0.85,
    'bicycle': 0.7,
    'airplane': 3.5,
    'boat': 2.0,
    'chair': 0.55,
    'couch': 2.2,
    'bed': 1.4,
    'dining table': 1.8,
    'tv': 1.2,
    'laptop': 0.35,
    'cell phone': 0.08,
    'traffic light': 0.3,
    'stop sign': 0.6,
    'parking meter': 0.2,
    'fire hydrant': 0.3,
    'sports ball': 0.22,
    'baseball bat': 0.07,
    'tennis racket': 0.35,
    'skateboard': 0.2,
}

OBJECT_SIZES_ENHANCED = {
    'person': {
        'width': 0.45,
        'confidence': 0.95,
        'min_width': 0.35,
        'max_width': 0.55,
        'height_ratio': 2.5
    },
    'car': {
        'width': 1.85,
        'confidence': 0.90,
        'min_width': 1.65,
        'max_width': 2.05,
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

DEFAULT_OBJECT_INFO = {
    'width': 0.5,
    'confidence': 0.7,
    'min_width': 0.3,
    'max_width': 0.8,
    'height_ratio': 1.5
}

FACE_ENCODINGS_PATH = Path("face_data/encodings.pkl")
FACE_METADATA_PATH = Path("face_data/metadata.json")
Path("face_data").mkdir(exist_ok=True)

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_metadata = {}
        self.face_detection_interval = 5
        self.frame_counter = 0
        self.last_seen_faces = {}
        self.recognition_threshold = 0.6
        self.pending_new_face = None
        self.pending_new_encoding = None
        self.last_alert_time = {}
        self.alert_cooldown = 60
        self.load_face_data()

    def load_face_data(self):
        try:
            if FACE_ENCODINGS_PATH.exists():
                with open(FACE_ENCODINGS_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                print(f"[FACE] Loaded {len(self.known_face_names)} known faces")
            if FACE_METADATA_PATH.exists():
                with open(FACE_METADATA_PATH, 'r') as f:
                    self.known_face_metadata = json.load(f)
        except Exception as e:
            print(f"[FACE ERROR] Failed to load face data: {e}")

    def save_face_data(self):
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(FACE_ENCODINGS_PATH, 'wb') as f:
                pickle.dump(data, f)
            with open(FACE_METADATA_PATH, 'w') as f:
                json.dump(self.known_face_metadata, f, indent=2)
            print(f"[FACE] Saved {len(self.known_face_names)} face(s) to disk")
        except Exception as e:
            print(f"[FACE ERROR] Failed to save face data: {e}")

    def process_frame(self, frame):
        self.frame_counter += 1
        if self.frame_counter % self.face_detection_interval != 0:
            return []
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            name, confidence = self.recognize_face(face_encoding)
            if name == "Unknown":
                face_results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (top, right, bottom, left),
                    'is_new': True,
                    'encoding': face_encoding
                })
            else:
                self.update_last_seen(name)
                face_results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (top, right, bottom, left),
                    'is_new': False,
                    'encoding': None
                })
        return face_results

    def recognize_face(self, face_encoding):
        if len(self.known_face_encodings) == 0:
            return "Unknown", 0.0
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        best_distance = face_distances[best_match_index]
        confidence = 1.0 - best_distance
        if best_distance <= self.recognition_threshold:
            name = self.known_face_names[best_match_index]
            return name, confidence
        else:
            return "Unknown", confidence

    def add_new_face(self, face_encoding, name):
        if not name or name.strip() == "":
            print("[FACE] Invalid name provided")
            return False
        name = name.strip()
        if name in self.known_face_names:
            print(f"[FACE] Name '{name}' already exists")
            return False
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        self.known_face_metadata[name] = {
            'added_date': datetime.now().isoformat(),
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'times_seen': 1
        }
        self.save_face_data()
        print(f"[FACE] Added new face: {name}")
        return True

    def update_last_seen(self, name):
        if name in self.known_face_metadata:
            current_time = datetime.now().isoformat()
            self.known_face_metadata[name]['last_seen'] = current_time
            self.known_face_metadata[name]['times_seen'] += 1
            self.last_seen_faces[name] = current_time

    def should_alert(self, name):
        current_time = time.time()
        if name not in self.last_alert_time:
            self.last_alert_time[name] = current_time
            return True
        time_since_last_alert = current_time - self.last_alert_time[name]
        if time_since_last_alert >= self.alert_cooldown:
            self.last_alert_time[name] = current_time
            return True
        return False

    def get_alert_message(self, name):
        if name in self.known_face_metadata:
            metadata = self.known_face_metadata[name]
            times_seen = metadata.get('times_seen', 1)
            return f"Alert! {name} is in front of you."
        return f"Alert! {name} is in front of you."

    def delete_face(self, name):
        if name in self.known_face_names:
            index = self.known_face_names.index(name)
            self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)
            if name in self.known_face_metadata:
                del self.known_face_metadata[name]
            self.save_face_data()
            print(f"[FACE] Deleted face: {name}")
            return True
        return False

class SafetyNavigationSystem:
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        self.danger_zone_radius = 100
        self.last_warning_time = 0
        self.warning_cooldown = 2.0
        self.debug_mode = False
        self.last_spoken_alerts = set()
        self.alert_cooldown = 3.0
        self.calibration_points = []
        self.focal_length_calibrated = False
        self.actual_focal_length = FOCAL_LENGTH
        self.object_tracking_history = {}
        self.frame_count = 0
        self.motion_vectors = {}
        self.previous_frame_objects = []
        self.camera_moving = False
        self.movement_threshold = 5.0
        self.distance_confidence_history = {}
        self.min_confidence_threshold = 0.3
        self.calibration_quality = 0.0
        self.calibration_samples = 0
        self.required_calibration_samples = 5

    def toggle_debug(self):
        self.debug_mode = not self.debug_mode
        print(f"[DEBUG] Debug mode {'ON' if self.debug_mode else 'OFF'}")

    def get_debug_info(self):
        return f"Debug Mode: {'ON' if self.debug_mode else 'OFF'}"

    def should_speak_alert(self, alert_text):
        import time
        current_time = time.time()
        alert_key = alert_text[:50]
        if alert_key in self.last_spoken_alerts:
            return False
        self.last_spoken_alerts.add(alert_key)
        if current_time - self.last_warning_time > self.alert_cooldown:
            self.last_spoken_alerts.clear()
            self.last_warning_time = current_time
        return True

    def get_object_direction(self, relative_x):
        if relative_x > 0.3:
            return "to your right"
        elif relative_x < -0.3:
            return "to your left"
        else:
            return "in front of you"

    def calibrate_distance(self, known_distance_meters, object_width_pixels, object_type):
        if object_type in OBJECT_SIZES:
            known_width = OBJECT_SIZES[object_type]
            calculated_focal = (object_width_pixels * known_distance_meters) / known_width
            calibration_point = {
                'focal_length': calculated_focal,
                'distance': known_distance_meters,
                'object_type': object_type,
                'confidence': 0.8,
                'timestamp': time.time()
            }
            self.calibration_points.append(calibration_point)
            self.calibration_samples += 1
            if self.calibration_samples >= 3:
                focal_lengths = [cp['focal_length'] for cp in self.calibration_points[-5:]]
                median_focal = np.median(focal_lengths)
                focal_std = np.std(focal_lengths)
                focal_mean = np.mean(focal_lengths)
                if focal_mean > 0:
                    self.calibration_quality = 1.0 / (1.0 + (focal_std / focal_mean))
                else:
                    self.calibration_quality = 0.0
                if self.calibration_quality > 0.7:
                    self.actual_focal_length = median_focal
                    self.focal_length_calibrated = True
                    print(f"[CALIBRATION] High quality focal length: {median_focal:.1f}px (quality: {self.calibration_quality:.2f})")
                else:
                    self.actual_focal_length = focal_mean
                    self.focal_length_calibrated = True
                    print(f"[CALIBRATION] Standard focal length: {focal_mean:.1f}px (quality: {self.calibration_quality:.2f})")
                if self.debug_mode:
                    print(f"[DEBUG] Calibration samples: {self.calibration_samples}, Std: {focal_std:.1f}, Quality: {self.calibration_quality:.2f}")
            else:
                self.actual_focal_length = calculated_focal
                self.focal_length_calibrated = True
                print(f"[CALIBRATION] Initial focal length: {calculated_focal:.1f}px (samples: {self.calibration_samples})")
            return True
        return False

    def estimate_distance_advanced(self, object_width_pixels, object_type, confidence=0.5, object_id=None):
        if object_width_pixels == 0:
            return float('inf')
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
        distance1 = (known_width * self.actual_focal_length) / object_width_pixels
        min_distance = (min_width * self.actual_focal_length) / object_width_pixels
        max_distance = (max_width * self.actual_focal_length) / object_width_pixels
        if distance1 < min_distance or distance1 > max_distance:
            confidence *= 0.7
        distance2 = float('inf')
        if object_id and object_id in self.object_tracking_history:
            tracking_info = self.get_object_tracking_info(object_id)
            if tracking_info and tracking_info['frames_tracked'] >= 3:
                recent_distances = []
                for det in self.object_tracking_history[object_id][-3:]:
                    if det['width'] > 0:
                        recent_distances.append((known_width * self.actual_focal_length) / det['width'])
                if recent_distances:
                    distance2 = np.mean(recent_distances)
                    stability_factor = tracking_info['size_stability']
                    if stability_factor > 0.8:
                        confidence *= 1.2
                    elif stability_factor < 0.5:
                        confidence *= 0.8
        distance3 = float('inf')
        if object_id and object_id in self.motion_vectors and self.camera_moving:
            motion_info = self.motion_vectors[object_id]
            if motion_info['magnitude'] > 2.0:
                motion_factor = motion_info['magnitude'] / 10.0
                distance3 = max(0.5, 5.0 / (1.0 + motion_factor))
                if self.debug_mode:
                    print(f"[DEBUG] Motion parallax distance: {distance3:.2f}m (motion: {motion_factor:.2f})")
        distance4 = float('inf')
        if hasattr(self, 'last_detected_objects') and len(self.last_detected_objects) > 1:
            for obj in self.last_detected_objects:
                if obj['label'] == object_type and obj['distance'] > 0 and obj['distance'] < 20:
                    size_ratio = object_width_pixels / obj.get('pixel_width', object_width_pixels)
                    if size_ratio > 0.1 and size_ratio < 10:
                        distance4 = obj['distance'] * size_ratio
                        break
        distance5 = float('inf')
        if self.calibration_quality > 0.8:
            distance5 = (known_width * self.actual_focal_length) / object_width_pixels
        else:
            adaptive_focal = self.actual_focal_length * (0.9 + 0.2 * confidence)
            distance5 = (known_width * adaptive_focal) / object_width_pixels
        distances = []
        weights = []
        if distance1 != float('inf'):
            distances.append(distance1)
            weights.append(0.4)
        if distance2 != float('inf'):
            distances.append(distance2)
            weights.append(0.25)
        if distance3 != float('inf'):
            distances.append(distance3)
            weights.append(0.15)
        if distance4 != float('inf'):
            distances.append(distance4)
            weights.append(0.15)
        if distance5 != float('inf'):
            distances.append(distance5)
            weights.append(0.05)
        if len(distances) > 0:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            weighted_distance = sum(d * w for d, w in zip(distances, weights))
            confidence_factor = 0.8 + (confidence * 0.4)
            final_distance = weighted_distance * confidence_factor
            final_distance *= (0.9 + size_confidence * 0.2)
            final_distance = max(0.1, min(final_distance, 50.0))
            if self.debug_mode and len(distances) > 1:
                print(f"[DEBUG] Distance methods: {len(distances)}, Final: {final_distance:.2f}m, Confidence: {confidence:.2f}")
            return final_distance
        return distance1

    def estimate_distance(self, object_width_pixels, object_type, confidence=0.5):
        return self.estimate_distance_advanced(object_width_pixels, object_type, confidence)

    def calculate_object_position(self, x1, y1, x2, y2):
        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2
        relative_x = (object_center_x - self.center_x) / self.center_x
        relative_y = (object_center_y - self.center_y) / self.center_y
        distance_from_center = np.sqrt(relative_x ** 2 + relative_y ** 2)
        return relative_x, relative_y, distance_from_center

    def is_in_danger_zone(self, x1, y1, x2, y2, distance):
        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2
        center_distance = np.sqrt((object_center_x - self.center_x) ** 2 + (object_center_y - self.center_y) ** 2)
        return distance < 2.0 and center_distance < self.danger_zone_radius

    def get_direction_instruction(self, relative_x, relative_y):
        if abs(relative_x) < 0.3:
            if relative_y > 0.3:
                return "Obstacle directly ahead, stop immediately"
            else:
                return "Obstacle ahead, proceed with caution"
        elif relative_x > 0.3:
            return "Obstacle to your right, move left"
        else:
            return "Obstacle to your left, move right"

    def detect_crosswalk(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=100, maxLineGap=8)
        if lines is None:
            if self.debug_mode:
                print("[DEBUG] No lines detected in crosswalk detection")
            return False
        horizontal_lines = 0
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 12 and abs(x2 - x1) > 80:
                line_y = (y1 + y2) / 2
                if line_y > self.frame_height * 0.4:
                    horizontal_lines += 1
                    valid_lines.append(line[0])
        if self.debug_mode:
            print(f"[DEBUG] Crosswalk detection: {horizontal_lines} horizontal lines found")
        if horizontal_lines >= 4:
            if len(valid_lines) >= 4:
                y_coords = []
                for line in valid_lines:
                    y_coords.append((line[1] + line[3]) / 2)
                y_coords.sort()
                spacings = []
                for i in range(1, len(y_coords)):
                    spacings.append(y_coords[i] - y_coords[i - 1])
                if len(spacings) >= 2:
                    avg_spacing = sum(spacings) / len(spacings)
                    consistent_spacing = all(abs(spacing - avg_spacing) < avg_spacing * 0.3 for spacing in spacings)
                    if consistent_spacing and 15 <= avg_spacing <= 60:
                        if self.debug_mode:
                            print(f"[DEBUG] CROSSWALK DETECTED: {horizontal_lines} lines, avg spacing={avg_spacing:.1f}px")
                        return True
        return False

    def generate_safety_alert(self, object_type, distance, relative_x, relative_y, is_dangerous):
        if is_dangerous:
            direction = self.get_direction_instruction(relative_x, relative_y)
            return f"WARNING! {object_type} {distance:.1f} meters away. {direction}"
        elif distance < SAFETY_DISTANCES.get(object_type, SAFETY_DISTANCES['default']):
            direction = self.get_direction_instruction(relative_x, relative_y)
            return f"Caution: {object_type} {distance:.1f} meters ahead. {direction}"
        else:
            return f"{object_type} detected {distance:.1f} meters away"

    def suggest_path(self, detected_objects):
        if not detected_objects:
            return "Path ahead is clear, you can proceed safely"
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
        if len(center_obstacles) == 0:
            return "Path ahead is clear, proceed straight"
        elif len(left_obstacles) < len(right_obstacles):
            return "Obstacles ahead, consider moving to your left"
        else:
            return "Obstacles ahead, consider moving to your right"

    def track_object_across_frames(self, object_id, bbox, label, confidence):
        current_time = time.time()
        if object_id not in self.object_tracking_history:
            self.object_tracking_history[object_id] = []
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
        if len(self.object_tracking_history[object_id]) > 10:
            self.object_tracking_history[object_id] = self.object_tracking_history[object_id][-10:]
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
            if abs(motion_x) > self.movement_threshold or abs(motion_y) > self.movement_threshold:
                self.camera_moving = True
            else:
                self.camera_moving = False
        return len(self.object_tracking_history[object_id])

    def get_object_tracking_info(self, object_id):
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
        if object_id in self.object_tracking_history:
            history = self.object_tracking_history[object_id]
            if len(history) >= 3:
                confidences = [det['confidence'] for det in history[-3:]]
                if len(confidences) >= 2:
                    return confidences[-1] - confidences[0]
        return 0.0

safety_system = SafetyNavigationSystem()
face_system = FaceRecognitionSystem()

last_command = ""
lock = threading.Lock()
current_speech_rate = "+20%"
voice_thread_running = False
last_safety_command_time = 0
pending_face_name_input = False

def listen_thread():
    global last_command
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    with sr.Microphone() as source:
        print("[VOICE] Microphone initialized and listening...")
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
    try:
        asyncio.run(speak_async(text, rate))
    except Exception as e:
        print("[TTS RUN ERROR]", e)

def read_text_with_openai(frame):
    if openai_client is None:
        print("[ERROR] OpenAI client is not available. Check API key and initialization.")
        return "Sorry, the text reading feature is currently unavailable."
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        print("[OPENAI] Sending frame to OpenAI for text recognition...")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant for accessibility. Only transcribe visible text from the provided image. Do not identify, describe, or analyze any people, faces, or personal attributes. Do not provide opinions, descriptions, or summariesâ€”output text only. Donot reply with sorry i cant assist you.make sure you read the "
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Read the text which is available in the image ignore people and any other objects in the frame. JUST READ dont give any judgements and dont any dimension and other things only give text. if you find any sign boards please mention it also"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        recognized_text = response.choices[0].message.content.strip()
        print(f"[OPENAI] Recognized text: '{recognized_text}'")
        if not recognized_text or "no text" in recognized_text.lower():
            return "I don't see any text in front of you."
        return f"{recognized_text}"
    except Exception as e:
        print(f"[OPENAI ERROR] An error occurred while processing the image: {e}")
        return "Sorry, I encountered an error while trying to read the text."

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not accessible")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("[INFO] Smart AI Assistant with Safety Navigation and Face Recognition Active")
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
    detected_objects = []
    safety_alerts = []
    stairs_detected = False
    crosswalk_detected = safety_system.detect_crosswalk(frame)
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        label = model.names[int(cls)]
        if label in obstacle_labels:
            object_width = x2 - x1
            distance = safety_system.estimate_distance(object_width, label)
            relative_x, relative_y, center_distance = safety_system.calculate_object_position(x1, y1, x2, y2)
            is_dangerous = safety_system.is_in_danger_zone(x1, y1, x2, y2, distance)
            detected_objects.append({
                'label': label,
                'distance': distance,
                'relative_x': relative_x,
                'relative_y': relative_y,
                'is_dangerous': is_dangerous
            })
            alert = safety_system.generate_safety_alert(label, distance, relative_x, relative_y, is_dangerous)
            if safety_system.should_speak_alert(alert):
                safety_alerts.append(alert)
                if is_dangerous and label in AUTO_WARNING_OBJECTS:
                    direction = safety_system.get_object_direction(relative_x)
                    speak(f"WARNING! {label} dangerously close, {direction}!", current_speech_rate)
            color = (0, 0, 255) if is_dangerous else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {distance:.1f}m", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.line(frame, (int(x1 + object_width / 2), int(y2)),
                     (safety_system.center_x, safety_system.frame_height), (255, 255, 0), 1)

    face_results = face_system.process_frame(frame)

    for face_info in face_results:
        top, right, bottom, left = face_info['location']
        name = face_info['name']
        confidence = face_info['confidence']
        is_new = face_info['is_new']
        if is_new:
            color = (0, 165, 255)
            label_text = "Unknown - Say 'add face'"
        else:
            color = (0, 255, 0)
            label_text = f"{name} ({confidence * 100:.1f}%)"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label_text, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        if is_new and face_info['encoding'] is not None:
            face_system.pending_new_face = face_info['location']
            face_system.pending_new_encoding = face_info['encoding']
        if not is_new and name != "Unknown":
            if face_system.should_alert(name):
                alert_message = face_system.get_alert_message(name)
                speak(alert_message, current_speech_rate)

    cv2.circle(frame, (safety_system.center_x, safety_system.frame_height),
               safety_system.danger_zone_radius, (0, 255, 255), 2)

    if stairs_detected:
        cv2.putText(frame, "STAIRS DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        stairs_alert = "Stairs detected ahead, be careful with elevation change"
        safety_alerts.append(stairs_alert)
        if safety_system.should_speak_alert(stairs_alert):
            speak(stairs_alert, current_speech_rate)

    if crosswalk_detected:
        cv2.putText(frame, "CROSSWALK", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        crosswalk_alert = "Crosswalk detected, check for traffic signals"
        safety_alerts.append(crosswalk_alert)

    if not voice_thread_running:
        voice_thread = threading.Thread(target=listen_thread, daemon=True)
        voice_thread.start()
        voice_thread_running = True
        print("[VOICE] Voice recognition thread started")

    with lock:
        if last_command:
            print(f"[DEBUG] Processing command: '{last_command}'")

        if ("who" in last_command and "front" in last_command) or ("who" in last_command and "in front" in last_command):
            print("[DEBUG] 'who is in front' command detected")
            if not face_results:
                speak("I don't see anyone in front of you.", current_speech_rate)
                last_command = ""
            else:
                seen_people = []
                for face_info in face_results:
                    if face_info['name'] != "Unknown":
                        seen_people.append(face_info['name'])
                    else:
                        seen_people.append("an unknown person")
                seen_people = list(dict.fromkeys(seen_people))
                if len(seen_people) == 1:
                    message = f"I see {seen_people[0]} in front of you."
                else:
                    if "an unknown person" in seen_people:
                        others = [n for n in seen_people if n != "an unknown person"]
                        if others:
                            message = "I see " + ", ".join(others) + " and an unknown person in front of you."
                        else:
                            message = "I see an unknown person in front of you."
                    else:
                        message = "I see " + ", ".join(seen_people) + " in front of you."
                print("[FACE RESPONSE]", message)
                speak(message, current_speech_rate)
                last_command = ""

        elif "add" in last_command and "face" in last_command:
            print("[DEBUG] 'add face' command detected")
            if face_system.pending_new_encoding is not None:
                speak("Please say the person's name", current_speech_rate)
                pending_face_name_input = True
                last_command = ""
            else:
                speak("No unknown face detected. Please wait for someone to appear.", current_speech_rate)
                last_command = ""

        elif pending_face_name_input and last_command:
            print(f"[DEBUG] Name input: '{last_command}'")
            name = last_command.strip()
            if face_system.add_new_face(face_system.pending_new_encoding, name):
                speak(f"Successfully added {name} to the database", current_speech_rate)
            else:
                speak(f"Failed to add face. Name may already exist.", current_speech_rate)
            pending_face_name_input = False
            face_system.pending_new_face = None
            face_system.pending_new_encoding = None
            last_command = ""

        elif "list" in last_command and "faces" in last_command:
            print("[DEBUG] 'list faces' command detected")
            if len(face_system.known_face_names) > 0:
                names = ", ".join(face_system.known_face_names)
                message = f"I know {len(face_system.known_face_names)} people: {names}"
            else:
                message = "I don't know anyone yet"
            speak(message, current_speech_rate)
            last_command = ""

        elif "delete" in last_command and "face" in last_command:
            print("[DEBUG] 'delete face' command detected")
            speak("Say the name of the person to delete", current_speech_rate)
            last_command = ""

        elif "what" in last_command and "front" in last_command:
            print("[DEBUG] 'what front' command detected")
            if detected_objects:
                message = "I see: " + ", ".join(
                    [
                        f"a {obj['label']} {safety_system.get_object_direction(obj['relative_x'])} about {obj['distance']:.1f} meters away"
                        for obj in detected_objects])
                if any(obj['is_dangerous'] for obj in detected_objects):
                    message += ". WARNING: Some objects are dangerously close!"
            else:
                message = "Path ahead is clear, no obstacles detected."
            path_suggestion = safety_system.suggest_path(detected_objects)
            message += f" {path_suggestion}"
            print("[RESPONSE]", message)
            speak(message, current_speech_rate)
            last_command = ""

        elif "safety" in last_command or "danger" in last_command:
            current_time = time.time()
            if current_time - last_safety_command_time > 2.0:
                print("[DEBUG] 'safety/danger' command detected")
                if safety_alerts:
                    message = "Safety alerts: " + ". ".join(safety_alerts[:1])
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

        elif ("read" in last_command and "text" in last_command) or "sign boards" in last_command:
            print("[DEBUG] 'read text' command detected")
            speak("Looking for text...", current_speech_rate)
            text_to_read = read_text_with_openai(frame)
            speak(text_to_read, current_speech_rate)
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
            try:
                if current_speech_rate.startswith("+"):
                    current_rate = int(current_speech_rate[1:-1])
                    new_rate = min(current_rate + 10, 100)
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
            try:
                if current_speech_rate.startswith("+"):
                    current_rate = int(current_speech_rate[1:-1])
                    new_rate = max(current_rate - 10, 10)
                    current_speech_rate = f"+{new_rate}%"
                else:
                    current_rate = int(current_speech_rate[1:-1])
                    new_rate = max(current_rate - 10, -50)
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
                safety_system.alert_cooldown = 3.0
                print("[AUTO] Automatic critical warnings enabled")
                speak("Automatic critical warnings enabled", current_speech_rate)
            elif "off" in last_command or "disable" in last_command:
                safety_system.alert_cooldown = 999999
                print("[AUTO] Automatic critical warnings disabled")
                speak("Automatic critical warnings disabled", current_speech_rate)
            else:
                if safety_system.alert_cooldown > 1000:
                    safety_system.alert_cooldown = 3.0
                    print("[AUTO] Automatic critical warnings enabled")
                    speak("Automatic critical warnings enabled", current_speech_rate)
                else:
                    safety_system.alert_cooldown = 999999
                    print("[AUTO] Automatic critical warnings disabled")
                    speak("Automatic critical warnings disabled", current_speech_rate)
            last_command = ""

        elif "test" in last_command or "hello" in last_command:
            print("[DEBUG] 'test/hello' command detected")
            message = "Voice recognition is working! I can hear you clearly."
            print("[TEST]", message)
            speak(message, current_speech_rate)
            last_command = ""

        elif last_command:
            print(f"[DEBUG] Unrecognized command: '{last_command}'")
            last_command = ""

    y_offset = 110
    for i, alert in enumerate(safety_alerts[:3]):
        cv2.putText(frame, alert[:50] + "...", (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    speed_text = f"Speech Speed: {current_speech_rate}"
    cv2.putText(frame, speed_text, (10, y_offset + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    auto_status = "Critical Warnings: AUTO" if safety_system.alert_cooldown < 1000 else "Critical Warnings: OFF"
    auto_color = (0, 255, 0) if safety_system.alert_cooldown < 1000 else (0, 0, 255)
    cv2.putText(frame, auto_status, (10, y_offset + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, auto_color, 2)

    voice_status = f"Voice: {last_command[:30]}" if last_command else "Voice: Listening..."
    voice_color = (0, 255, 0) if last_command else (255, 255, 0)
    cv2.putText(frame, voice_status, (10, y_offset + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)

    face_count = len(face_system.known_face_names)
    face_status = f"Known Faces: {face_count}"
    cv2.putText(frame, face_status, (10, y_offset + 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    commands_hint = "Say: 'what front', 'add face', 'list faces'"
    cv2.putText(frame, commands_hint, (10, y_offset + 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Smart AI Assistant - Safety Navigation + Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
