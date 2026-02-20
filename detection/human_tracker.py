"""
Advanced Human Detection and ID Tracking with Memory-Based Occlusion Handling
Uses DeepSORT-inspired algorithm for persistent tracking with JSON persistence
"""

import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import time
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple, Optional
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Import memory manager
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.person_memory_manager import PersonMemoryManager

class HumanTracker:
    def __init__(self, model_path="yolov8n.pt", camera_index=0):
        """
        Initialize Advanced Human Tracker with Memory-Based Occlusion Handling
        
        Args:
            model_path: Path to YOLOv8 model
            camera_index: Camera index (default 0 for webcam)
        """
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        print(f"✓ Advanced Human Tracker Model loaded: {model_path}")
        
        # Camera setup
        self.camera_index = camera_index
        self.cap = None
        
        # Advanced tracking parameters
        self.frame_count = 0
        self.next_person_id = 1
        
        # Memory system for occlusion handling
        self.person_memory = {}  # Long-term memory
        self.active_tracks = {}  # Currently visible tracks
        self.lost_tracks = {}  # Recently lost tracks (for occlusion)
        
        # Initialize JSON memory manager
        self.memory_manager = PersonMemoryManager()
        
        # Tracking parameters
        self.max_disappeared_frames = 30  # 1 second at 30fps
        self.max_memory_time = 300  # 5 minutes memory
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4  # Intersection over Union for matching
        self.feature_threshold = 0.7  # Feature similarity threshold
        self.max_disappeared_time_seconds = 10  # Delete ID after 10 seconds
        
        # Colors for visualization (consistent colors per ID)
        self.id_colors = {}  # Persistent colors per ID
    
    def extract_features(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract appearance features from person region
        
        Args:
            frame: Input frame
            bbox: [x1, y1, w, h] bounding box
            
        Returns:
            Feature vector
        """
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        # Ensure bbox is within frame bounds
        x1 = max(0, min(x1, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))
        
        # Extract person region
        person_region = frame[y1:y2, x1:x2]
        
        if person_region.size == 0:
            return np.zeros(128)  # Default feature
        
        # Resize to standard size
        try:
            person_region = cv2.resize(person_region, (32, 32))
            
            # Extract color histogram features
            hist_b = cv2.calcHist([person_region], [0], [None], [16], [0, 256])
            hist_g = cv2.calcHist([person_region], [1], [None], [16], [0, 256])
            hist_r = cv2.calcHist([person_region], [2], [None], [16], [0, 256])
            
            # Normalize histograms
            hist_b = hist_b.flatten() / (hist_b.sum() + 1e-8)
            hist_g = hist_g.flatten() / (hist_g.sum() + 1e-8)
            hist_r = hist_r.flatten() / (hist_r.sum() + 1e-8)
            
            # Combine features
            features = np.concatenate([hist_b, hist_g, hist_r])
            
            return features
            
        except Exception:
            return np.zeros(128)
    
    def calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections_to_tracks(self, detections: List[Dict]) -> List[Tuple]:
        """
        Match current detections to existing tracks using Hungarian algorithm
        
        Returns:
            List of (detection_idx, track_idx, match_score) tuples
        """
        matches = []
        
        for det_idx, detection in enumerate(detections):
            det_bbox = detection['bbox']
            det_features = detection.get('features', None)
            
            best_match = None
            best_score = 0
            
            for track_id, track in self.active_tracks.items():
                track_bbox = track['bbox']
                
                # Calculate IoU
                iou = self.calculate_iou(det_bbox, track_bbox)
                
                # Calculate feature similarity if available
                feature_sim = 0
                if det_features is not None and 'features' in track:
                    try:
                        feature_sim = 1 - cosine(det_features, track['features'])
                        feature_sim = max(0, feature_sim)
                    except:
                        feature_sim = 0
                
                # Combined score
                combined_score = 0.6 * iou + 0.4 * feature_sim
                
                if combined_score > self.iou_threshold and combined_score > best_score:
                    best_match = (det_idx, track_id, combined_score)
                    best_score = combined_score
            
            if best_match:
                matches.append(best_match)
        
        return matches
    
    def detect_humans(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Advanced human detection with memory-based occlusion handling and JSON persistence
        
        Args:
            frame: Input frame
            
        Returns:
            List of human detection dictionaries
        """
        # Update frame count
        self.frame_count += 1
        
        # Detect humans using YOLO
        results = self.model(frame, stream=True, conf=self.confidence_threshold)
        current_detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class and confidence
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                # Only process person detections
                if class_name.lower() == 'person':
                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    # Extract features
                    features = self.extract_features(frame, [x1, y1, w, h])
                    
                    detection = {
                        'bbox': [x1, y1, w, h],
                        'confidence': conf,
                        'features': features,
                        'center': (x1 + w // 2, y1 + h // 2)
                    }
                    
                    current_detections.append(detection)
        
        # Match detections to existing tracks
        matches = self.match_detections_to_tracks(current_detections)
        
        # Update matched tracks
        matched_detections = set()
        matched_tracks = set()
        
        for det_idx, track_id, score in matches:
            detection = current_detections[det_idx]
            track = self.active_tracks[track_id]
            
            # Update track with new detection
            self.active_tracks[track_id] = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'features': detection['features'],
                'center': detection['center'],
                'last_seen': self.frame_count,
                'disappeared_count': 0,
                'age': track.get('age', 0) + 1
            }
            
            # Save to memory manager
            self.memory_manager.add_or_update_person(
                track_id, 
                detection['features'], 
                detection['bbox'], 
                detection['confidence'],
                self.frame_count,
                self.camera_index
            )
            
            matched_detections.add(det_idx)
            matched_tracks.add(track_id)
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(current_detections):
            if det_idx not in matched_detections:
                # Try to find matching person from memory (JSON persistence)
                matched_id = self.memory_manager.find_matching_person(
                    detection['features'],
                    detection['bbox'],
                    detection['confidence'],
                    self.max_disappeared_time_seconds
                )
                
                if matched_id:
                    # Use existing ID from memory
                    person_id = matched_id
                    print(f"🔄 Recovered person ID {person_id} from JSON memory")
                else:
                    # Assign new ID
                    person_id = self.next_person_id
                    self.next_person_id += 1
                    print(f"➕ Assigned new person ID {person_id}")
                
                self.active_tracks[person_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'features': detection['features'],
                    'center': detection['center'],
                    'last_seen': self.frame_count,
                    'disappeared_count': 0,
                    'age': 0
                }
                
                # Save to memory manager
                is_new_person = self.memory_manager.add_or_update_person(
                    person_id, 
                    detection['features'], 
                    detection['bbox'], 
                    detection['confidence'],
                    self.frame_count,
                    self.camera_index
                )
                
                if is_new_person:
                    print(f"🆕 Added new person {person_id} to JSON memory")
        
        # Handle unmatched tracks (potential occlusion)
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matched_tracks:
                track = self.active_tracks[track_id]
                track['disappeared_count'] = track.get('disappeared_count', 0) + 1
                
                # If track has been missing for too long, move to lost tracks
                if track['disappeared_count'] > self.max_disappeared_frames:
                    # Mark as inactive in memory
                    self.memory_manager.mark_person_inactive(track_id, "occlusion")
                    
                    # Move to lost tracks for potential recovery
                    self.lost_tracks[track_id] = {
                        **track,
                        'lost_time': self.frame_count,
                        'lost_timestamp': time.time()  # Add actual timestamp
                    }
                    del self.active_tracks[track_id]
                else:
                    # Keep in active tracks but update
                    self.active_tracks[track_id] = track
        
        # Try to recover lost tracks
        self.recover_lost_tracks(current_detections)
        
        # Clean old memory
        self.memory_manager.cleanup_old_memory()
        
        # Save memory to JSON file periodically
        if self.frame_count % 30 == 0:  # Save every 30 frames
            self.memory_manager.force_save()
        
        # Convert to detection format
        final_detections = []
        for track_id, track in self.active_tracks.items():
            x1, y1, w, h = track['bbox']
            
            detection = {
                "id": track_id,
                "bbox": [x1, y1, w, h],
                "person_conf": track['confidence'],
                "gun_conf": 0.0,
                "knife_conf": 0.0,
                "fight_conf": 0.0,
                "meta": {
                    "class_name": "PERSON",
                    "raw_confidence": track['confidence'],
                    "frame": self.frame_count,
                    "camera": self.camera_index
                },
                "timestamp": time.time(),
                "frame": frame.copy()
            }
            
            final_detections.append(detection)
        
        return final_detections
    
    def recover_lost_tracks(self, current_detections: List[Dict]):
        """
        Try to recover lost tracks using memory-based matching
        
        Args:
            current_detections: Current frame detections
        """
        if not self.lost_tracks:
            return
        
        recovered_tracks = []
        
        for lost_id, lost_track in list(self.lost_tracks.items()):
            # Check if enough time has passed for recovery attempt
            if self.frame_count - lost_track['lost_time'] > 10:  # Try recovery after 10 frames
                continue
            
            # Try to match with current detections
            best_match = None
            best_score = 0
            
            for detection in current_detections:
                # Skip if already matched
                if detection.get('matched', False):
                    continue
                
                # Calculate IoU with predicted position
                iou = self.calculate_iou(detection['bbox'], lost_track['bbox'])
                
                # Calculate feature similarity
                feature_sim = 0
                if 'features' in detection and 'features' in lost_track:
                    try:
                        feature_sim = 1 - cosine(detection['features'], lost_track['features'])
                        feature_sim = max(0, feature_sim)
                    except:
                        feature_sim = 0
                
                # Combined score with higher weight on features for occlusion recovery
                combined_score = 0.3 * iou + 0.7 * feature_sim
                
                if combined_score > self.feature_threshold and combined_score > best_score:
                    best_match = (detection, combined_score)
                    best_score = combined_score
            
            # If good match found, recover the track
            if best_match and best_score > 0.6:
                detection, score = best_match
                detection['matched'] = True  # Mark as matched
                
                # Recover track
                self.active_tracks[lost_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'features': detection['features'],
                    'center': detection['center'],
                    'last_seen': self.frame_count,
                    'disappeared_count': 0,
                    'age': lost_track.get('age', 0) + 1
                }
                
                # Update memory
                self.person_memory[lost_id] = {
                    'features': detection['features'],
                    'last_seen': self.frame_count,
                    'bbox_history': lost_track.get('bbox_history', []) + [detection['bbox']][-10:],
                    'confidence_history': lost_track.get('confidence_history', []) + [detection['confidence']][-5:]
                }
                
                recovered_tracks.append(lost_id)
                del self.lost_tracks[lost_id]
                print(f"✓ Recovered track ID {lost_id} after occlusion (score: {best_score:.2f})")
        
        # Remove very old lost tracks
        current_time = self.frame_count
        self.lost_tracks = {
            tid: track for tid, track in self.lost_tracks.items()
            if current_time - track['lost_time'] < 100  # Keep for ~3 seconds
        }
    
    def cleanup_memory(self):
        """Clean up old memory entries and delete IDs after 10 seconds"""
        current_time = self.frame_count
        current_timestamp = time.time()
        
        # Clean person memory (keep for 5 minutes)
        self.person_memory = {
            pid: memory for pid, memory in self.person_memory.items()
            if current_time - memory['last_seen'] < self.max_memory_time
        }
        
        # Delete lost tracks that have been missing for more than 10 seconds
        deleted_ids = []
        for tid, track in list(self.lost_tracks.items()):
            # Calculate time disappeared in seconds
            disappeared_time_seconds = (current_timestamp - track.get('lost_timestamp', current_timestamp))
            
            if disappeared_time_seconds > self.max_disappeared_time_seconds:
                deleted_ids.append(tid)
                del self.lost_tracks[tid]
                
                # Remove from person memory as well
                if tid in self.person_memory:
                    del self.person_memory[tid]
                
                # Remove color assignment
                if tid in self.id_colors:
                    del self.id_colors[tid]
                
                print(f"✗ Deleted track ID {tid} after {disappeared_time_seconds:.1f} seconds disappearance")
        
        # Clean very old lost tracks (backup cleanup)
        self.lost_tracks = {
            tid: track for tid, track in self.lost_tracks.items()
            if current_time - track['lost_time'] < 100  # Keep for ~3 seconds max
        }
    
    def get_id_color(self, person_id: int) -> Tuple[int, int, int]:
        """Get consistent color for person ID"""
        if person_id not in self.id_colors:
            # Generate consistent color based on ID
            np.random.seed(person_id)  # Seed with ID for consistency
            self.id_colors[person_id] = (
                np.random.randint(50, 255),  # Avoid too dark colors
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            )
        
        return self.id_colors[person_id]
    
    def update_frame_count(self):
        """Update frame counter"""
        self.frame_count += 1
