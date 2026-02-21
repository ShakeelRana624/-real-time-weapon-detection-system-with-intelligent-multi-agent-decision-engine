"""
Fight Detection Module for Intelligent Weapon Detection System

Detects fighting/fighting behavior using fd_v3.h5 model and integrates with person tracking.
Shows fight detection with red bounding boxes and person IDs.
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Tuple, Optional
import time
import os

class FightDetector:
    """Fight detection using fd_v3.h5 model"""
    
    def __init__(self, model_path: str = "models/fd_v3.h5"):
        """
        Initialize fight detector
        
        Args:
            model_path: Path to fight detection model
        """
        try:
            # Load the fight detection model
            self.model = tf.keras.models.load_model(model_path)
            print(f"✓ Fight model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load fight model: {e}")
            self.model = None
        
        # Fight detection parameters
        self.confidence_threshold = 0.7
        self.fight_sequence_length = 30  # Number of frames to analyze
        self.fight_detection_frames = []  # Buffer for fight detection
        
        # Storage for fight data
        self.detected_fights = {}  # {person_id: fight_info}
        self.fight_active = False
        self.last_fight_time = 0
        
        # Model input size (adjust based on your model requirements)
        self.input_size = (224, 224)  # Common size for fight detection models
        
    def preprocess_frame_for_fight_detection(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Preprocess frame region for fight detection model
        
        Args:
            frame: Input frame
            bbox: Bounding box [x, y, w, h]
            
        Returns:
            Preprocessed frame ready for model input
        """
        try:
            x, y, w, h = bbox
            
            # Extract person region
            person_region = frame[y:y+h, x:x+w]
            
            if person_region.size == 0:
                return None
            
            # Resize to model input size
            person_region = cv2.resize(person_region, self.input_size)
            
            # Normalize pixel values (0-1 range)
            person_region = person_region.astype(np.float32) / 255.0
            
            # Add batch dimension
            person_region = np.expand_dims(person_region, axis=0)
            
            return person_region
            
        except Exception as e:
            print(f"Error preprocessing frame for fight detection: {e}")
            return None
    
    def detect_fight_in_region(self, frame: np.ndarray, bbox: List[int]) -> Tuple[bool, float]:
        """
        Detect fight in a specific region
        
        Args:
            frame: Input frame
            bbox: Bounding box [x, y, w, h]
            
        Returns:
            Tuple: (is_fight, confidence)
        """
        if self.model is None:
            return False, 0.0
        
        try:
            # Preprocess the region
            processed_region = self.preprocess_frame_for_fight_detection(frame, bbox)
            
            if processed_region is None:
                return False, 0.0
            
            # Predict fight
            prediction = self.model.predict(processed_region, verbose=0)
            
            # Get fight probability (assuming binary classification: 0=no_fight, 1=fight)
            fight_probability = float(prediction[0][1]) if len(prediction[0]) > 1 else float(prediction[0][0])
            
            # Determine if fight detected
            is_fight = fight_probability > self.confidence_threshold
            
            return is_fight, fight_probability
            
        except Exception as e:
            print(f"Error in fight detection: {e}")
            return False, 0.0
    
    def detect_fights_in_frame(self, frame: np.ndarray, person_detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Detect fights for all persons in frame
        
        Args:
            frame: Input frame
            person_detections: List of person detections with bounding boxes and IDs
            
        Returns:
            Dictionary: {person_id: fight_info}
        """
        if self.model is None:
            return {}
        
        fight_results = {}
        current_time = time.time()
        
        try:
            # Detect fights for each person
            for person_detection in person_detections:
                person_id = person_detection.get("id")
                person_bbox = person_detection.get("bbox", [])
                
                if not person_id or len(person_bbox) < 4:
                    continue
                
                # Detect fight in person's region
                is_fight, confidence = self.detect_fight_in_region(frame, person_bbox)
                
                # Store fight detection
                fight_info = {
                    "person_id": person_id,
                    "fight_detected": is_fight,
                    "confidence": confidence,
                    "bbox": person_bbox,
                    "timestamp": current_time
                }
                
                fight_results[person_id] = fight_info
                
                # Update storage
                self.detected_fights[person_id] = fight_info
                
                # Print fight detection info
                if is_fight:
                    print(f"🥊 FIGHT DETECTED: Person {person_id} (confidence: {confidence:.2f})")
                    
                    # Update global fight status
                    self.fight_active = True
                    self.last_fight_time = current_time
                
                elif person_id in self.detected_fights and self.detected_fights[person_id].get("fight_detected", False):
                    # Fight was detected before but not now
                    print(f"✅ FIGHT ENDED: Person {person_id}")
        
        except Exception as e:
            print(f"Error in fight detection: {e}")
        
        # Check if any fight is still active
        if not any(info.get("fight_detected", False) for info in fight_results.values()):
            if self.fight_active:
                print("✅ ALL FIGHTS ENDED")
                self.fight_active = False
        
        return fight_results
    
    def get_fight_info(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get fight information for a specific person
        
        Args:
            person_id: Person ID
            
        Returns:
            Fight information or None
        """
        return self.detected_fights.get(person_id)
    
    def get_all_fights(self) -> Dict[int, Dict[str, Any]]:
        """Get all detected fights"""
        return self.detected_fights.copy()
    
    def get_fight_count(self) -> int:
        """Get count of persons currently fighting"""
        count = 0
        for fight_info in self.detected_fights.values():
            if fight_info.get("fight_detected", False):
                count += 1
        return count
    
    def get_fighting_person_ids(self) -> List[int]:
        """Get list of person IDs currently fighting"""
        ids = []
        for person_id, fight_info in self.detected_fights.items():
            if fight_info.get("fight_detected", False):
                ids.append(person_id)
        return ids
    
    def is_fight_active(self) -> bool:
        """Check if any fight is currently active"""
        return self.fight_active
    
    def clear_fights(self):
        """Clear all stored fights"""
        self.detected_fights.clear()
        self.fight_active = False
        self.fight_detection_frames.clear()
    
    def draw_fight_on_frame(self, frame: np.ndarray, fight_info: Dict[str, Any]) -> np.ndarray:
        """
        Draw fight detection on frame
        
        Args:
            frame: Input frame
            fight_info: Fight information dictionary
            color: Color for drawing
            
        Returns:
            Frame with fight detection drawn
        """
        try:
            person_id = fight_info.get("person_id")
            fight_detected = fight_info.get("fight_detected", False)
            confidence = fight_info.get("confidence", 0.0)
            bbox = fight_info.get("bbox", [])
            
            if not bbox or len(bbox) < 4:
                return frame
            
            x, y, w, h = bbox[:4]
            
            if fight_detected:
                # Draw red bounding box for fight
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                
                # Draw fight label
                fight_label = f"🥊 FIGHT ID:{person_id}"
                conf_text = f"Conf:{confidence:.2f}"
                
                # Draw label background
                label_size = cv2.getTextSize(fight_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x, y - 40), (x + label_size[0] + 10, y - 10), (0, 0, 255), -1)
                
                # Draw label text
                cv2.putText(frame, fight_label, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, conf_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Add warning indicators
                cv2.putText(frame, "⚠️ VIOLENCE DETECTED", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing fight detection: {e}")
            return frame
    
    def update_fight_statistics(self, stats: Dict[str, Any]):
        """Update statistics with fight detection information"""
        if "fight_detections" not in stats:
            stats["fight_detections"] = 0
        
        current_fight_count = self.get_fight_count()
        if current_fight_count > stats.get("fight_detections", 0):
            stats["fight_detections"] = current_fight_count
