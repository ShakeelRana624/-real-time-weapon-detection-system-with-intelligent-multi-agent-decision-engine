"""
Complete Integrated Gun Detection System

Combines YOLO gun detection with agent-based decision making
for real-time threat assessment and response.
"""

import cv2
import time
import numpy as np
import threading
import sqlite3
from typing import Dict, List, Any, Optional
from ultralytics import YOLO
import platform
from datetime import datetime
import json

# Import agent-based decision engine
from agent_based_decision_engine import AgentBasedDecisionEngine

class IntegratedGunDetectionSystem:
    """Complete integrated system with YOLO detection and agent-based decision making"""
    
    def __init__(self, model_path: str = "best.pt", camera_index: int = 0):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        print(f"✓ Model loaded: {model_path}")
        
        # Initialize agent-based decision engine
        self.decision_engine = AgentBasedDecisionEngine()
        print("✓ Agent-based decision engine initialized")
        
        # Get reference to evidence agent for direct frame buffering
        self.evidence_agent = self.decision_engine.evidence_agent
        print("✓ Evidence agent connected for frame buffering")
        
        # Camera setup
        self.camera_index = camera_index
        self.cap = None
        
        # Tracking system
        self.person_id_counter = 0
        self.active_tracks = {}
        self.frame_count = 0
        
        # Evidence storage
        self.evidence_folder = "evidence"
        self.init_evidence_storage()
        
        # Alert system
        self.alert_active = False
        self.last_alert_time = 0
        
        # Statistics
        self.stats = {
            "total_detections": 0,
            "threat_detections": 0,
            "alerts_triggered": 0,
            "evidence_saved": 0
        }
    
    def init_evidence_storage(self):
        """Initialize evidence storage folder and database"""
        import os
        os.makedirs(self.evidence_folder, exist_ok=True)
        
        # Initialize database
        self.db_path = os.path.join(self.evidence_folder, "detections.db")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                detection_id TEXT,
                bbox TEXT,
                confidence REAL,
                threat_level TEXT,
                threat_score REAL,
                actions TEXT,
                evidence_path TEXT,
                frame_data BLOB
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✓ Evidence storage initialized: {self.evidence_folder}")
    
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✓ Camera {self.camera_index} started")
        return True
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects using YOLO model"""
        results = self.model(frame, stream=True, conf=0.5)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detection = self.convert_yolo_to_detection(box, frame)
                if detection:
                    detections.append(detection)
        
        return detections
    
    def convert_yolo_to_detection(self, box, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Convert YOLO detection to agent format"""
        try:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            
            # Confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self.model.names[cls]
            
            # Create or update track
            person_id = self.update_track(x1, y1, w, h, class_name, conf)
            
            # Build detection dict
            detection = {
                "id": person_id,
                "bbox": [x1, y1, w, h],
                "person_conf": conf if class_name.lower() == "gun" else 0.0,
                "gun_conf": conf if class_name.lower() == "gun" else 0.0,
                "knife_conf": conf if class_name.lower() == "knife" else 0.0,
                "fight_conf": 0.0,  # Would need separate model
                "meta": {
                    "class_name": class_name,
                    "raw_confidence": conf,
                    "frame": self.frame_count,
                    "camera": self.camera_index
                },
                "timestamp": time.time(),
                "frame": frame.copy()
            }
            
            return detection
            
        except Exception as e:
            print(f"Error converting detection: {e}")
            return None
    
    def update_track(self, x: int, y: int, w: int, h: int, 
                   class_name: str, conf: float) -> int:
        """Update or create person tracks"""
        center_x, center_y = x + w//2, y + h//2
        
        # Find existing track
        best_match_id = None
        best_distance = float('inf')
        
        for track_id, track_info in self.active_tracks.items():
            track_center = track_info["center"]
            distance = np.sqrt((center_x - track_center[0])**2 + 
                             (center_y - track_center[1])**2)
            
            if distance < 100 and distance < best_distance:
                best_distance = distance
                best_match_id = track_id
        
        if best_match_id is None:
            # Create new track
            self.person_id_counter += 1
            best_match_id = self.person_id_counter
        
        # Update track
        self.active_tracks[best_match_id] = {
            "center": (center_x, center_y),
            "bbox": [x, y, w, h],
            "last_seen": self.frame_count,
            "class_name": class_name,
            "confidence": conf
        }
        
        # Clean old tracks
        self.clean_old_tracks()
        
        return best_match_id
    
    def clean_old_tracks(self):
        """Remove tracks not seen recently"""
        current_frame = self.frame_count
        stale_ids = []
        
        for track_id, track_info in self.active_tracks.items():
            if current_frame - track_info["last_seen"] > 30:
                stale_ids.append(track_id)
        
        for stale_id in stale_ids:
            del self.active_tracks[stale_id]
    
    def process_detections(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process detections through agent-based decision engine"""
        results = []
        
        # Add frame to evidence buffer for all frames (not just weapon detections)
        self.evidence_agent.add_frame_to_buffer(frame, time.time())
        
        for detection in detections:
            # Process through agent engine
            result = self.decision_engine.process(detection)
            results.append(result)
            
            # Update statistics
            self.stats["total_detections"] += 1
            if result["threat_score"] > 1.0:  # Lowered threshold for testing
                self.stats["threat_detections"] += 1
            
            # Save evidence if needed (traditional image evidence)
            if "SAVE_EVIDENCE" in result["action"]:
                self.save_evidence(detection, result, frame)
            
            # Trigger alerts if needed
            if "LOCAL_ALARM" in result["action"]:
                self.trigger_alert(result)
            
            # Save to database
            self.save_to_database(detection, result, frame)
        
        return results
    
    def save_evidence(self, detection: Dict[str, Any], result: Dict[str, Any], frame: np.ndarray):
        """Save evidence frame"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evidence_{detection['id']}_{timestamp}.jpg"
            filepath = f"{self.evidence_folder}/{filename}"
            
            # Draw annotations on frame
            annotated_frame = self.draw_annotations(frame, detection, result)
            cv2.imwrite(filepath, annotated_frame)
            
            self.stats["evidence_saved"] += 1
            print(f"✓ Evidence saved: {filename}")
            
        except Exception as e:
            print(f"Error saving evidence: {e}")
    
    def trigger_alert(self, result: Dict[str, Any]):
        """Trigger alert system"""
        current_time = time.time()
        
        # Prevent alert spam (minimum 2 seconds between alerts)
        if current_time - self.last_alert_time < 2:
            return
        
        self.last_alert_time = current_time
        self.stats["alerts_triggered"] += 1
        
        # Play alert sound
        self.play_alert_sound(result["state"])
        
        # Print alert
        print(f"🚨 ALERT TRIGGERED: {result['state']} (Score: {result['threat_score']:.2f})")
    
    def play_alert_sound(self, threat_level: str):
        """Play alert sound based on threat level"""
        try:
            if platform.system() == "Windows":
                import winsound
                if threat_level in ["CRITICAL", "VIOLENT"]:
                    winsound.Beep(1500, 500)  # High frequency, longer
                elif threat_level in ["HIGH", "ARMED"]:
                    winsound.Beep(1000, 300)  # Medium frequency
                else:
                    winsound.Beep(800, 200)   # Lower frequency
            else:
                print('\a')  # System bell for Unix
        except Exception as e:
            print(f"Alert sound failed: {e}")
    
    def save_to_database(self, detection: Dict[str, Any], result: Dict[str, Any], frame: np.ndarray):
        """Save detection to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Compress frame for storage
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_data = buffer.tobytes()
            
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, detection_id, bbox, confidence, threat_level, threat_score, actions, evidence_path, frame_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection["timestamp"],
                detection["id"],
                json.dumps(detection["bbox"]),
                detection.get("person_conf", 0),
                result["state"],
                result["threat_score"],
                result["action"],
                f"evidence_{detection['id']}.jpg",
                frame_data
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Database error: {e}")
    
    def draw_annotations(self, frame: np.ndarray, detection: Dict[str, Any], result: Dict[str, Any]) -> np.ndarray:
        """Draw annotations on frame"""
        annotated = frame.copy()
        
        # Bounding box
        bbox = detection["bbox"]
        x, y, w, h = bbox
        
        # Get color based on system state
        system_state = result.get("system_state", "normal").upper()
        color = self.get_system_state_color(system_state)
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        
        # Label with system state
        label = f"ID:{detection['id']} {system_state}"
        score_text = f"Score:{result['threat_score']:.1f}"
        
        # Background for label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(annotated, (x, y - 50), (x + label_size[0], y), color, -1)
        
        # Text
        cv2.putText(annotated, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, score_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add timestamp and system state
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"State: {system_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add state change indicator
        if result.get("state_changed", False):
            cv2.putText(annotated, "STATE CHANGED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return annotated
    
    def get_system_state_color(self, state: str) -> tuple:
        """Get color based on system state"""
        colors = {
            "NORMAL": (0, 255, 0),      # Green
            "SUSPICIOUS": (0, 255, 255), # Yellow
            "THREAT_DETECTION": (0, 165, 255), # Orange
            "EMERGENCY": (0, 0, 255),    # Red
            # Legacy compatibility
            "MINIMAL": (0, 255, 0),
            "LOW": (255, 255, 0),
            "MEDIUM": (255, 165, 0),
            "HIGH": (255, 0, 0),
            "CRITICAL": (255, 0, 0),
            "VIOLENT": (0, 0, 255),
            "ARMED": (255, 165, 0)
        }
        return colors.get(state, (255, 255, 255))
    
    def draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """Draw statistics on frame"""
        stats_frame = frame.copy()
        
        # Stats background
        cv2.rectangle(stats_frame, (10, 60), (350, 180), (0, 0, 0), -1)
        
        # Get current system state
        current_state = getattr(self.decision_engine.state_agent, 'state_transition', None)
        system_state = current_state.current_state.value if current_state else "Unknown"
        
        # Stats text
        stats_text = [
            f"Total Detections: {self.stats['total_detections']}",
            f"Threat Detections: {self.stats['threat_detections']}",
            f"Alerts Triggered: {self.stats['alerts_triggered']}",
            f"Evidence Saved: {self.stats['evidence_saved']}",
            f"System State: {system_state.upper()}"
        ]
        
        for i, text in enumerate(stats_text):
            color = (0, 255, 0) if i < 4 else self.get_system_state_color(system_state.upper())
            cv2.putText(stats_frame, text, (20, 80 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return stats_frame
    
    def run(self):
        """Main detection loop"""
        if not self.start_camera():
            return
        
        print("\n=== INTEGRATED GUN DETECTION SYSTEM ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'r' to reset statistics")
        print("Press 'e' to view evidence folder")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera read error")
                    break
                
                self.frame_count += 1
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Process detections
                results = self.process_detections(detections, frame)
                
                # Draw results
                for detection, result in zip(detections, results):
                    frame = self.draw_annotations(frame, detection, result)
                
                # Draw statistics
                frame = self.draw_stats(frame)
                
                # Display frame
                cv2.imshow("Integrated Gun Detection System", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_manual_frame(frame)
                elif key == ord('r'):
                    self.reset_statistics()
                elif key == ord('e'):
                    self.open_evidence_folder()
        
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        finally:
            self.cleanup()
    
    def save_manual_frame(self, frame: np.ndarray):
        """Manually save current frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_capture_{timestamp}.jpg"
        filepath = f"{self.evidence_folder}/{filename}"
        cv2.imwrite(filepath, frame)
        print(f"✓ Manual frame saved: {filename}")
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            "total_detections": 0,
            "threat_detections": 0,
            "alerts_triggered": 0,
            "evidence_saved": 0
        }
        print("✓ Statistics reset")
    
    def open_evidence_folder(self):
        """Open evidence folder"""
        import subprocess
        import os
        try:
            if platform.system() == "Windows":
                os.startfile(self.evidence_folder)
            else:
                subprocess.run(["xdg-open", self.evidence_folder])
            print(f"✓ Evidence folder opened: {self.evidence_folder}")
        except Exception as e:
            print(f"Could not open evidence folder: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        # Stop evidence recording
        if hasattr(self, 'evidence_agent'):
            self.evidence_agent.force_stop_recording()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System shutdown complete")
        
        # Print final statistics
        print("\n=== FINAL STATISTICS ===")
        for key, value in self.stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Print evidence agent status
        if hasattr(self, 'evidence_agent'):
            evidence_status = self.evidence_agent.get_status()
            print(f"\n=== EVIDENCE AGENT STATUS ===")
            print(f"Buffered Frames: {evidence_status['buffered_frames']}")
            print(f"Final Recording: {evidence_status['current_file'] or 'None'}")
            print(f"Total Recordings: Check evidence/videos/ folder")

def main():
    """Main entry point"""
    print("=" * 60)
    print("INTEGRATED GUN DETECTION SYSTEM")
    print("=" * 60)
    
    # Check if model exists
    import os
    if not os.path.exists("best.pt"):
        print("❌ Error: best.pt model not found!")
        print("Please ensure best.pt is in the current directory")
        return
    
    try:
        # Initialize and run system
        system = IntegratedGunDetectionSystem()
        system.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
