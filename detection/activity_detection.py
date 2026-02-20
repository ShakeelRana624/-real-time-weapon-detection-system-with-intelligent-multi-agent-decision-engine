# detection/activity_detection.py

import cv2
import numpy as np
from ultralytics import YOLO
import math
from enum import Enum


class HumanActivity(Enum):
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    RUNNING = "running"
    UNKNOWN = "unknown"


class ActivityDetector:
    """
    Advanced Activity detection using YOLOv8 Pose model
    (SITTING / STANDING / WALKING / RUNNING)
    """

    def __init__(self, model_path="yolov8n-pose.pt"):
        self.pose_model = YOLO(model_path)
        self.prev_leg_dist = {}
        
        # Enhanced pose tracking
        self.pose_history = {}
        self.frame_count = 0
        
        # COCO keypoint indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points in degrees"""
        if np.any(p1 == 0) or np.any(p2 == 0) or np.any(p3 == 0):
            return 0
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def detect_activity_enhanced(self, person_id, keypoints, confidences):
        """Enhanced activity detection with temporal analysis"""
        
        # Store current pose in history
        if person_id not in self.pose_history:
            self.pose_history[person_id] = []
        
        current_pose = {
            'left_ankle': keypoints[self.LEFT_ANKLE].copy() if confidences[self.LEFT_ANKLE] > 0.5 else np.array([0, 0]),
            'right_ankle': keypoints[self.RIGHT_ANKLE].copy() if confidences[self.RIGHT_ANKLE] > 0.5 else np.array([0, 0]),
            'left_knee': keypoints[self.LEFT_KNEE].copy() if confidences[self.LEFT_KNEE] > 0.5 else np.array([0, 0]),
            'right_knee': keypoints[self.RIGHT_KNEE].copy() if confidences[self.RIGHT_KNEE] > 0.5 else np.array([0, 0]),
            'left_hip': keypoints[self.LEFT_HIP].copy() if confidences[self.LEFT_HIP] > 0.5 else np.array([0, 0]),
            'right_hip': keypoints[self.RIGHT_HIP].copy() if confidences[self.RIGHT_HIP] > 0.5 else np.array([0, 0]),
            'frame': self.frame_count
        }
        
        self.pose_history[person_id].append(current_pose)
        
        # Keep only last 10 frames for analysis
        if len(self.pose_history[person_id]) > 10:
            self.pose_history[person_id].pop(0)
        
        # Need at least 3 frames for temporal analysis
        if len(self.pose_history[person_id]) < 3:
            return HumanActivity.UNKNOWN, 0.0
        
        # Calculate knee angles
        left_knee_angle = self.calculate_angle(
            keypoints[self.LEFT_HIP], keypoints[self.LEFT_KNEE], keypoints[self.LEFT_ANKLE]
        )
        right_knee_angle = self.calculate_angle(
            keypoints[self.RIGHT_HIP], keypoints[self.RIGHT_KNEE], keypoints[self.RIGHT_ANKLE]
        )
        
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # Standing detection
        if 160 <= avg_knee_angle <= 180:
            # Check for movement (walking/running)
            movement = self.calculate_movement(person_id)
            if movement > 20:  # High movement = running
                return HumanActivity.RUNNING, 0.8
            elif movement > 5:  # Moderate movement = walking
                return HumanActivity.WALKING, 0.7
            else:  # No movement = standing
                return HumanActivity.STANDING, 0.8
        
        # Sitting detection
        elif 70 <= avg_knee_angle <= 120:
            return HumanActivity.SITTING, 0.8
        
        return HumanActivity.UNKNOWN, 0.0
    
    def calculate_movement(self, person_id):
        """Calculate movement intensity from pose history"""
        if person_id not in self.pose_history or len(self.pose_history[person_id]) < 2:
            return 0.0
        
        poses = self.pose_history[person_id]
        total_movement = 0.0
        movement_count = 0
        
        for i in range(1, len(poses)):
            prev_pose = poses[i-1]
            curr_pose = poses[i]
            
            # Calculate ankle movement
            if np.any(prev_pose['left_ankle'] != 0) and np.any(curr_pose['left_ankle'] != 0):
                left_movement = np.linalg.norm(curr_pose['left_ankle'] - prev_pose['left_ankle'])
                total_movement += left_movement
                movement_count += 1
            
            if np.any(prev_pose['right_ankle'] != 0) and np.any(curr_pose['right_ankle'] != 0):
                right_movement = np.linalg.norm(curr_pose['right_ankle'] - prev_pose['right_ankle'])
                total_movement += right_movement
                movement_count += 1
        
        return total_movement / movement_count if movement_count > 0 else 0.0

    def _classify_activity(self, keypoints, track_id):
        """
        Original activity classification method (kept for compatibility)
        keypoints: numpy array (17, 2)
        """
        try:
            # COCO keypoints
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_knee = keypoints[13]
            right_knee = keypoints[14]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]

            # -------------------------------
            # CASE 1: HIP NOT VISIBLE → SITTING
            # -------------------------------
            if left_hip[0] == 0 and right_hip[0] == 0:
                return "SITTING"

            # -------------------------------
            # HIP CENTER
            # -------------------------------
            hip_center = np.array([
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            ])

            activity = "STANDING"  # default when hip visible

            # -------------------------------
            # WALKING DETECTION (movement-based)
            # -------------------------------
            movement_score = 0

            if track_id in self.prev_leg_dist:
                prev = self.prev_leg_dist[track_id]

                hip_move = np.linalg.norm(hip_center - prev["hip"])
                ankle_dist = np.linalg.norm(left_ankle - right_ankle)

                movement_score = hip_move + ankle_dist

                if movement_score > 25:
                    activity = "WALKING"

            # Store previous frame data
            self.prev_leg_dist[track_id] = {
                "hip": hip_center,
                "ankle_dist": np.linalg.norm(left_ankle - right_ankle)
            }

            return activity

        except Exception:
            return "SITTING"

    def detect(self, frame, humans):
        """
        Enhanced detection method with both original and new activity detection
        
        Input:
            frame
            humans = [
                {
                    "id": int,
                    "bbox": (x1, y1, x2, y2)
                }
            ]

        Output:
            activities = [
                {
                    "human_id": int,
                    "activity": str,
                    "risk_level": str
                }
            ]
        """
        self.frame_count += 1
        activities = []

        if not humans:
            return activities

        results = self.pose_model(frame, conf=0.4, verbose=False, imgsz=256)

        if results[0].keypoints is None:
            return activities

        keypoints_all = results[0].keypoints.xy.cpu().numpy()
        confidences_all = results[0].keypoints.conf.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for human in humans:
            hx1, hy1, hx2, hy2 = human["bbox"]
            human_id = human["id"]

            # Match pose bbox with human bbox (IoU-free simple overlap)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box

                if (
                    x1 < hx2 and x2 > hx1 and
                    y1 < hy2 and y2 > hy1
                ):
                    keypoints = keypoints_all[i]
                    confidences = confidences_all[i]
                    
                    # Use enhanced activity detection
                    activity_enum, activity_confidence = self.detect_activity_enhanced(
                        human_id, keypoints, confidences
                    )
                    activity = activity_enum.value
                    
                    # Fallback to original method if needed
                    if activity == "unknown":
                        activity = self._classify_activity(keypoints, human_id)

                    # Enhanced risk mapping
                    if activity == "running":
                        risk = "high"
                    elif activity == "walking":
                        risk = "medium"
                    elif activity == "sitting":
                        risk = "low"
                    else:  # standing
                        risk = "low"

                    activities.append({
                        "human_id": human_id,
                        "activity": activity.upper(),
                        "risk_level": risk,
                        "confidence": activity_confidence
                    })

        return activities
