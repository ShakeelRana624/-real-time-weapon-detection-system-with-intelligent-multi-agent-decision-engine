import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from datetime import datetime
import random
from typing import List, Dict, Any

def load_model():
    """Load YOLOv8 model"""
    print("Loading YOLOv8 model...")
    try:
        return YOLO("yolov8n.pt")
    except:
        print("Downloading YOLOv8 model...")
        return YOLO("yolov8n.yaml")

def initialize_tracker():
    """Initialize DeepSort tracker with enhanced occlusion handling"""
    return DeepSort(
        max_age=100,        # Increased from 50 to 100 frames (~3 seconds at 30fps)
        n_init=3,           # Reduced from 5 to 3 for faster tracking
        max_iou_distance=0.7,  # IoU threshold for matching
        nn_budget=100,      # Feature memory budget
        embedder="mobilenet",  # Use mobilenet embedder
        embedder_gpu=False   # Use CPU to avoid numpy conflicts
    )

def get_color_for_id(track_id):
    """Generate consistent random color for each track ID"""
    random.seed(track_id)  # Seed with track ID for consistent colors
    color = (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )
    return color

def get_detections(frame, model, confidence_threshold=0.35):
    """Get person detections from frame with occlusion handling"""
    results = model(frame, conf=confidence_threshold)
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Only detect people (class 0)
                if box.cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Filter small detections (likely false positives)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Minimum area threshold (adjust based on camera distance)
                    min_area = 1000  # pixels
                    
                    if area > min_area:
                        detections.append(([x1, y1, width, height], conf, 'person'))
    
    return detections

class HumanTracker:
    """Human Tracker using DeepSort algorithm"""
    
    def __init__(self, model_path="yolov8n.pt", camera_index=0):
        """
        Initialize Human Tracker with DeepSort
        
        Args:
            model_path: Path to YOLOv8 model
            camera_index: Camera index (default 0 for webcam)
        """
        # Load YOLOv8 model
        self.model = load_model()
        print(f"✓ Human Tracker Model loaded: {model_path}")
        
        # Initialize DeepSort tracker
        self.tracker = initialize_tracker()
        print("✓ DeepSort tracker initialized")
        
        # Camera setup
        self.camera_index = camera_index
        self.cap = None
        
        # Tracking parameters
        self.frame_count = 0
        self.confidence_threshold = 0.35
        
        # Colors for visualization (consistent colors per ID)
        self.id_colors = {}  # Persistent colors per ID
    
    def detect_humans(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect humans using DeepSort tracking
        
        Args:
            frame: Input frame
            
        Returns:
            List of human detection dictionaries with IDs
        """
        # Update frame count
        self.frame_count += 1
        
        # Get detections from YOLO
        detections = get_detections(frame, self.model, self.confidence_threshold)
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Convert tracks to detection format
        final_detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1
            
            # Get track confidence
            confidence = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.5
            confidence = float(confidence) if confidence is not None else 0.5
            
            detection = {
                "id": track_id,
                "bbox": [x1, y1, w, h],
                "person_conf": confidence,
                "gun_conf": 0.0,
                "knife_conf": 0.0,
                "fight_conf": 0.0,
                "meta": {
                    "class_name": "PERSON",
                    "raw_confidence": confidence,
                    "frame": self.frame_count,
                    "camera": self.camera_index,
                    "time_since_update": track.time_since_update if hasattr(track, 'time_since_update') else 0,
                    "is_occluded": track.time_since_update > 0 if hasattr(track, 'time_since_update') else False
                },
                "timestamp": time.time(),
                "frame": frame.copy()
            }
            
            final_detections.append(detection)
        
        return final_detections
    
    def get_id_color(self, person_id: int) -> tuple:
        """Get consistent color for person ID"""
        if person_id not in self.id_colors:
            # Generate consistent color based on ID
            self.id_colors[person_id] = get_color_for_id(person_id)
        
        return self.id_colors[person_id]
    
    def update_frame_count(self):
        """Update frame counter"""
        self.frame_count += 1
    
    def draw_tracking_info(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw tracking information on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with tracking information drawn
        """
        annotated = frame.copy()
        
        # Draw information overlay
        current_count = len(detections)
        unique_ids = set([d["id"] for d in detections])
        unique_count = len(unique_ids)
        
        # Draw background for text
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
        
        # Draw text
        cv2.putText(annotated, f"Current People: {current_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Total Unique: {unique_count}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated, f"Time: {datetime.now().strftime('%H:%M:%S')}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw bounding boxes and IDs for each detection
        for detection in detections:
            track_id = detection["id"]
            x1, y1, w, h = detection["bbox"]
            x2, y2 = x1 + w, y1 + h
            
            # Get unique color for this person
            color = self.get_id_color(track_id)
            
            # Check if track is occluded
            is_occluded = detection["meta"].get("is_occluded", False)
            time_since_update = detection["meta"].get("time_since_update", 0)
            
            # Draw bounding box with unique color
            if is_occluded:
                # Draw dashed bounding box for occluded person
                dash_length = 10
                for i in range(x1, x2, dash_length * 2):
                    start_x = min(i, x2)
                    end_x = min(i + dash_length, x2)
                    cv2.line(annotated, (start_x, y1), (end_x, y1), color, 3)
                    cv2.line(annotated, (start_x, y2), (end_x, y2), color, 3)
                
                for i in range(y1, y2, dash_length * 2):
                    start_y = min(i, y2)
                    end_y = min(i + dash_length, y2)
                    cv2.line(annotated, (x1, start_y), (x1, end_y), color, 3)
                    cv2.line(annotated, (x2, start_y), (x2, end_y), color, 3)
            else:
                # Solid bounding box for visible person
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw ID and label
            if is_occluded:
                label = f"Person {track_id} - OCCLUDED ({time_since_update}f)"
                label_color = (0, 0, 255)  # Red for occluded
            else:
                label = f"Person {track_id}"
                label_color = color
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw colored background for label
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), label_color, -1)
            
            # Draw text in white for contrast
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if is_occluded:
                # Draw X for occluded person
                cv2.drawMarker(annotated, (center_x, center_y), label_color, 
                             cv2.MARKER_CROSS, 10, 3)
            else:
                # Draw filled circle for visible person
                cv2.circle(annotated, (center_x, center_y), 5, color, -1)
                cv2.circle(annotated, (center_x, center_y), 7, (255, 255, 255), 1)
        
        return annotated

def main():
    """Main function to run camera tracking"""
    print("🎯 Starting People Tracking System...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Initialize tracker
    tracker = HumanTracker()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera initialized successfully")
    print("📹 Starting live feed...")
    
    # Tracking variables
    unique_people = set()
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Detect and track humans
        detections = tracker.detect_humans(frame)
        
        # Draw tracking information
        frame = tracker.draw_tracking_info(frame, detections)
        
        # Update unique people set
        for detection in detections:
            unique_people.add(detection["id"])
        
        # Calculate FPS
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show frame
        cv2.imshow("People Tracking System", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Quitting...")
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"people_tracking_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Session Summary:")
    print(f"   Total Frames: {frame_count}")
    print(f"   Total Unique People: {len(unique_people)}")
    print(f"   Session Duration: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
