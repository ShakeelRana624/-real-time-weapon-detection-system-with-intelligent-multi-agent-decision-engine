"""
Pose Detection Module for Intelligent Weapon Detection System

Detects hand-up poses using YOLO pose model and integrates with person tracking.
Shows pose information with person IDs in bounding boxes.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Dict, List, Any, Tuple, Optional
import time

class PoseDetector:
    """Hand-up pose detection using YOLO pose model"""
    
    def __init__(self, model_path: str = "models/yolov8n-pose.pt"):
        """
        Initialize pose detector
        
        Args:
            model_path: Path to YOLO pose model
        """
        try:
            self.model = YOLO(model_path)
            print(f"✓ Pose model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load pose model: {e}")
            self.model = None
        
        # Pose detection parameters
        self.confidence_threshold = 0.5
        self.hand_up_threshold = 0.7  # Confidence for hands-up detection
        
        # Hand-up pose detection logic
        self.hand_up_keypoints = [5, 6, 7, 8, 9, 10]  # Shoulder, elbow, wrist keypoints
        
        # Storage for pose data
        self.detected_poses = {}  # {person_id: pose_info}
        
    def detect_hands_up_pose(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if person is making hands-up pose
        
        Args:
            keypoints: Pose keypoints from YOLO model
            
        Returns:
            Tuple: (is_hands_up, confidence)
        """
        if keypoints is None or len(keypoints) < 17:
            return False, 0.0
        
        try:
            # Extract key points for hands-up detection
            # Keypoints format: [x, y, confidence] for each of 17 keypoints
            # 5: Left shoulder, 6: Right shoulder, 7: Left elbow, 8: Right elbow
            # 9: Left wrist, 10: Right wrist
            
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_elbow = keypoints[7]
            right_elbow = keypoints[8]
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            
            # Check if keypoints are detected (confidence > 0)
            if (left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3 or 
                left_elbow[2] < 0.3 or right_elbow[2] < 0.3 or
                left_wrist[2] < 0.3 or right_wrist[2] < 0.3):
                return False, 0.0
            
            # Calculate hands-up conditions
            hands_up_conditions = []
            
            # Condition 1: Wrists are above shoulders
            left_wrist_above_shoulder = left_wrist[1] < left_shoulder[1]
            right_wrist_above_shoulder = right_wrist[1] < right_shoulder[1]
            hands_up_conditions.append(left_wrist_above_shoulder and right_wrist_above_shoulder)
            
            # Condition 2: Wrists are above elbows
            left_wrist_above_elbow = left_wrist[1] < left_elbow[1]
            right_wrist_above_elbow = right_wrist[1] < right_elbow[1]
            hands_up_conditions.append(left_wrist_above_elbow and right_wrist_above_elbow)
            
            # Condition 3: Arms are relatively straight (elbows not too bent)
            left_arm_angle = self._calculate_angle(left_shoulder[:2], left_elbow[:2], left_wrist[:2])
            right_arm_angle = self._calculate_angle(right_shoulder[:2], right_elbow[:2], right_wrist[:2])
            
            # Arms should be mostly straight (angle > 120 degrees)
            left_arm_straight = left_arm_angle > 120
            right_arm_straight = right_arm_angle > 120
            hands_up_conditions.append(left_arm_straight and right_arm_straight)
            
            # Calculate confidence based on keypoint confidences and conditions
            avg_keypoint_confidence = np.mean([
                left_shoulder[2], right_shoulder[2], 
                left_elbow[2], right_elbow[2],
                left_wrist[2], right_wrist[2]
            ])
            
            # Count how many conditions are met
            conditions_met = sum(hands_up_conditions)
            condition_ratio = conditions_met / len(hands_up_conditions)
            
            # Final confidence
            hands_up_confidence = avg_keypoint_confidence * condition_ratio
            
            # Determine if hands-up pose
            is_hands_up = conditions_met >= 2 and hands_up_confidence > self.hand_up_threshold
            
            return is_hands_up, hands_up_confidence
            
        except Exception as e:
            print(f"Error in hands-up detection: {e}")
            return False, 0.0
    
    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Calculate angle between three points (point1-point2-point3)
        
        Args:
            point1: First point (shoulder)
            point2: Middle point (elbow)
            point3: Last point (wrist)
            
        Returns:
            Angle in degrees
        """
        try:
            # Calculate vectors
            v1 = point1 - point2
            v2 = point3 - point2
            
            # Calculate angle using dot product
            dot_product = np.dot(v1, v2)
            magnitude1 = np.linalg.norm(v1)
            magnitude2 = np.linalg.norm(v2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0.0
    
    def detect_poses_in_frame(self, frame: np.ndarray, person_detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Detect poses for all persons in frame
        
        Args:
            frame: Input frame
            person_detections: List of person detections with bounding boxes and IDs
            
        Returns:
            Dictionary: {person_id: pose_info}
        """
        if self.model is None:
            return {}
        
        pose_results = {}
        
        try:
            # Run pose detection on the entire frame
            results = self.model(frame, verbose=False)
            
            if not results or len(results) == 0:
                return pose_results
            
            # Get pose detections
            pose_detections = results[0]
            
            if pose_detections.keypoints is None:
                return pose_results
            
            # Match poses with person detections
            for person_detection in person_detections:
                person_id = person_detection.get("id")
                person_bbox = person_detection.get("bbox", [])
                
                if not person_id or len(person_bbox) < 4:
                    continue
                
                # Find the best matching pose for this person
                best_pose_match = self._find_best_pose_match(person_bbox, pose_detections)
                
                if best_pose_match is not None:
                    keypoints = best_pose_match
                    
                    # Detect hands-up pose
                    is_hands_up, confidence = self.detect_hands_up_pose(keypoints)
                    
                    pose_info = {
                        "person_id": person_id,
                        "pose_type": "HANDS_UP" if is_hands_up else "NORMAL",
                        "confidence": confidence,
                        "keypoints": keypoints.tolist(),
                        "bbox": person_bbox,
                        "timestamp": time.time()
                    }
                    
                    pose_results[person_id] = pose_info
                    
                    # Update storage
                    self.detected_poses[person_id] = pose_info
                    
                    print(f"🙋 Person {person_id}: {pose_info['pose_type']} (confidence: {confidence:.2f})")
        
        except Exception as e:
            print(f"Error in pose detection: {e}")
        
        return pose_results
    
    def _find_best_pose_match(self, person_bbox: List[int], pose_detections) -> Optional[np.ndarray]:
        """
        Find the best matching pose for a person bbox
        
        Args:
            person_bbox: Person bounding box [x, y, w, h]
            pose_detections: YOLO pose detection results
            
        Returns:
            Best matching keypoints or None
        """
        try:
            if pose_detections.keypoints is None:
                return None
            
            person_x, person_y, person_w, person_h = person_bbox
            person_center_x = person_x + person_w / 2
            person_center_y = person_y + person_h / 2
            
            best_match = None
            best_distance = float('inf')
            
            # Iterate through all detected poses
            for i, keypoints in enumerate(pose_detections.keypoints):
                if keypoints is None or len(keypoints.xy) == 0:
                    continue
                
                # Get pose keypoints (use nose or body center as reference)
                pose_keypoints = keypoints.xy[0].cpu().numpy()  # First person's keypoints
                
                if len(pose_keypoints) < 17:
                    continue
                
                # Calculate center of pose (use average of visible keypoints)
                visible_keypoints = pose_keypoints[keypoints.conf[0].cpu().numpy() > 0.3]
                
                if len(visible_keypoints) == 0:
                    continue
                
                pose_center_x = np.mean(visible_keypoints[:, 0])
                pose_center_y = np.mean(visible_keypoints[:, 1])
                
                # Calculate distance between person bbox center and pose center
                distance = np.sqrt((person_center_x - pose_center_x)**2 + 
                                 (person_center_y - pose_center_y)**2)
                
                # Check if pose is within person bbox (with some tolerance)
                tolerance = 50  # pixels
                if (abs(pose_center_x - person_center_x) < person_w/2 + tolerance and
                    abs(pose_center_y - person_center_y) < person_h/2 + tolerance):
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = keypoints.data[0].cpu().numpy()  # Convert to numpy
            
            return best_match
            
        except Exception as e:
            print(f"Error finding pose match: {e}")
            return None
    
    def get_pose_info(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get pose information for a specific person
        
        Args:
            person_id: Person ID
            
        Returns:
            Pose information or None
        """
        return self.detected_poses.get(person_id)
    
    def get_all_poses(self) -> Dict[int, Dict[str, Any]]:
        """Get all detected poses"""
        return self.detected_poses.copy()
    
    def clear_poses(self):
        """Clear all stored poses"""
        self.detected_poses.clear()
    
    def draw_pose_on_frame(self, frame: np.ndarray, pose_info: Dict[str, Any], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw pose skeleton and information on frame
        
        Args:
            frame: Input frame
            pose_info: Pose information dictionary
            color: Color for drawing
            
        Returns:
            Frame with pose drawn
        """
        try:
            keypoints = np.array(pose_info.get("keypoints", []))
            person_id = pose_info.get("person_id")
            pose_type = pose_info.get("pose_type", "NORMAL")
            confidence = pose_info.get("confidence", 0.0)
            bbox = pose_info.get("bbox", [])
            
            if len(keypoints) < 17:
                return frame
            
            # Draw pose skeleton
            skeleton_connections = [
                (5, 6), (5, 7), (7, 9),   # Left arm
                (6, 8), (8, 10),        # Right arm
                (5, 11), (6, 12),       # Torso
                (11, 12), (11, 13), (13, 15),  # Left leg
                (12, 14), (14, 15)       # Right leg
            ]
            
            # Draw connections
            for connection in skeleton_connections:
                pt1_idx, pt2_idx = connection
                if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                    keypoints[pt1_idx][2] > 0.3 and keypoints[pt2_idx][2] > 0.3):
                    
                    pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                    pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                    
                    cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw keypoints
            for i, keypoint in enumerate(keypoints):
                if keypoint[2] > 0.3:  # Only draw visible keypoints
                    pt = (int(keypoint[0]), int(keypoint[1]))
                    cv2.circle(frame, pt, 3, color, -1)
            
            # Draw pose label
            if bbox and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                
                # Create pose label
                pose_color = (0, 255, 0) if pose_type == "HANDS_UP" else (255, 255, 255)
                label = f"ID:{person_id} {pose_type}"
                conf_text = f"Conf:{confidence:.2f}"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - 40), (x + label_size[0], y - 10), pose_color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, conf_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, pose_color, 1)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing pose: {e}")
            return frame
    
    def get_hands_up_count(self) -> int:
        """Get count of persons with hands-up pose"""
        count = 0
        for pose_info in self.detected_poses.values():
            if pose_info.get("pose_type") == "HANDS_UP":
                count += 1
        return count
    
    def get_hands_up_person_ids(self) -> List[int]:
        """Get list of person IDs with hands-up pose"""
        ids = []
        for person_id, pose_info in self.detected_poses.items():
            if pose_info.get("pose_type") == "HANDS_UP":
                ids.append(person_id)
        return ids
