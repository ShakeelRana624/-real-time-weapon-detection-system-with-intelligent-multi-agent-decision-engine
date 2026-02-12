"""
Complete Integrated Gun Detection System

Combines YOLO gun detection with agent-based decision making
for real-time threat assessment and response.
"""

import cv2
import numpy as np
import time
import os
import platform
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from ultralytics import YOLO

# Import our agent-based decision engine
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
        """Draw professional annotations on frame"""
        annotated = frame.copy()
        
        # Bounding box with enhanced styling
        bbox = detection["bbox"]
        x, y, w, h = bbox
        
        # Get color based on system state
        system_state = result.get("system_state", "normal").upper()
        color = self.get_system_state_color(system_state)
        
        # Draw thick bounding box with glow effect
        for i in range(3):
            alpha = 255 - (i * 80)
            cv2.rectangle(annotated, (x-i, y-i), (x + w + i, y + h + i), 
                         (color[0], color[1], color[2]), 1)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        
        # Enhanced label with background
        label = f"ID:{detection['id']} {system_state}"
        score_text = f"Score:{result['threat_score']:.1f}"
        
        # Calculate label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        bg_height = 60
        cv2.rectangle(annotated, (x, y - bg_height), (x + label_size[0] + 20, y), color, -1)
        cv2.rectangle(annotated, (x, y - bg_height), (x + label_size[0] + 20, y), (255, 255, 255), 1)
        
        # Text with shadow effect
        cv2.putText(annotated, label, (x + 10, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, score_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add weapon confidence indicators
        confidences = []
        if detection.get("gun_conf", 0) > 0.1:
            confidences.append(f"🔫 Gun:{detection['gun_conf']:.2f}")
        if detection.get("knife_conf", 0) > 0.1:
            confidences.append(f"🔪 Knife:{detection['knife_conf']:.2f}")
        if detection.get("explosion_conf", 0) > 0.1:
            confidences.append(f"💥 Explosion:{detection['explosion_conf']:.2f}")
        if detection.get("grenade_conf", 0) > 0.1:
            confidences.append(f"🧨 Grenade:{detection['grenade_conf']:.2f}")
        
        # Draw confidence indicators
        if confidences:
            conf_text = " | ".join(confidences[:3])  # Limit to 3 items
            cv2.putText(annotated, conf_text, (x, y + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add state change indicator with animation effect
        if result.get("state_changed", False):
            cv2.putText(annotated, "⚡ STATE CHANGED!", (x, y + h + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add emergency indicator if active
        emergency_response = result.get("emergency_response")
        if emergency_response and not emergency_response.get("emergency_deactivated"):
            threat_type = emergency_response.get("threat_type", "UNKNOWN")
            cv2.putText(annotated, f"🚨 EMERGENCY: {threat_type}", 
                       (annotated.shape[1] // 2 - 150, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
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
        """Draw professional statistics on frame"""
        stats_frame = frame.copy()
        
        # Create professional overlay panels
        height, width = stats_frame.shape[:2]
        
        # Top header bar
        cv2.rectangle(stats_frame, (0, 0), (width, 60), (20, 20, 20), -1)
        cv2.rectangle(stats_frame, (0, 0), (width, 60), (0, 255, 0), 2)
        
        # System title
        cv2.putText(stats_frame, "INTELLIGENT WEAPON DETECTION SYSTEM", (20, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(stats_frame, "AI-Powered Security Monitoring", (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Right side status panel (smaller for 4-section layout)
        panel_width = 200
        cv2.rectangle(stats_frame, (width - panel_width, 60), (width, height - 40), (30, 30, 30), -1)
        cv2.rectangle(stats_frame, (width - panel_width, 60), (width, height - 40), (0, 255, 0), 2)
        
        # Get current system state
        current_state = getattr(self.decision_engine.state_agent, 'state_transition', None)
        system_state = current_state.current_state.value if current_state else "UNKNOWN"
        state_color = self.get_system_state_color(system_state.upper())
        
        # Status header
        cv2.putText(stats_frame, "STATUS", (width - panel_width + 10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw state indicator with color
        cv2.rectangle(stats_frame, (width - panel_width + 10, 90), 
                     (width - panel_width + 25, 105), state_color, -1)
        cv2.putText(stats_frame, f"{system_state[:4].upper()}", (width - panel_width + 30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 2)
        
        # Compact statistics
        stats_text = [
            f"Total: {self.stats['total_detections']}",
            f"Threat: {self.stats['threat_detections']}",
            f"Alerts: {self.stats['alerts_triggered']}",
            f"Evidence: {self.stats['evidence_saved']}",
            "━━━━━━━━━━━━",
            f"FPS: {30:.1f}",
        ]
        
        y_offset = 125
        for text in stats_text:
            color = (255, 255, 255) if text.startswith(("Total", "Threat", "Alerts", "Evidence")) else (100, 100, 100)
            if text.startswith("FPS:"):
                color = (0, 255, 0)
            cv2.putText(stats_frame, text, (width - panel_width + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_offset += 18
        
        # Bottom control bar
        cv2.rectangle(stats_frame, (0, height - 40), (width, height), (20, 20, 20), -1)
        cv2.rectangle(stats_frame, (0, height - 40), (width, height), (0, 255, 0), 1)
        
        # Controls info
        controls = "[Q] Quit [S] Save [R] Reset [E] Evidence"
        cv2.putText(stats_frame, controls, (20, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(stats_frame, timestamp, (width - 100, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return stats_frame
    
    def create_birds_eye_view(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Create bird's eye view of detection area"""
        # Create blank canvas for bird's eye view
        birds_eye = np.zeros((240, 320, 3), dtype=np.uint8)
        birds_eye.fill(20)  # Dark background
        
        # Draw grid
        for i in range(0, 320, 40):
            cv2.line(birds_eye, (i, 0), (i, 240), (40, 40, 40), 1)
        for i in range(0, 240, 40):
            cv2.line(birds_eye, (0, i), (320, i), (40, 40, 40), 1)
        
        # Draw detection area boundary
        cv2.rectangle(birds_eye, (10, 10), (310, 230), (0, 255, 0), 2)
        
        # Add title
        cv2.putText(birds_eye, "BIRD'S EYE VIEW", (80, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Plot detections as circles
        for detection in detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                # Convert to bird's eye view coordinates
                bird_x = int((x + w/2) * 320 / frame.shape[1])
                bird_y = int((y + h/2) * 240 / frame.shape[0])
                
                # Get threat level
                threat_score = detection.get("threat_score", 0)
                if threat_score > 2.0:
                    color = (0, 0, 255)  # Red for high threat
                elif threat_score > 1.0:
                    color = (0, 165, 255)  # Orange for medium threat
                else:
                    color = (0, 255, 0)  # Green for low threat
                
                # Draw detection point
                cv2.circle(birds_eye, (bird_x, bird_y), 8, color, -1)
                cv2.circle(birds_eye, (bird_x, bird_y), 10, color, 2)
                
                # Add ID
                cv2.putText(birds_eye, str(detection.get("id", "?")), 
                           (bird_x - 5, bird_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return birds_eye
    
    def create_enhanced_heatmap(self, frame: np.ndarray, detections_history: List[Dict[str, Any]]) -> np.ndarray:
        """Create enhanced activity heatmap with better visualization"""
        heatmap = np.zeros((180, 240, 3), dtype=np.uint8)
        heatmap.fill(20)  # Dark background
        
        # Add title
        cv2.putText(heatmap, "ACTIVITY HEATMAP", (50, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Create enhanced heatmap based on detection history
        if detections_history:
            # Draw fine grid
            grid_size = 20
            for i in range(0, 240, grid_size):
                cv2.line(heatmap, (i, 30), (i, 180), (30, 30, 30), 1)
            for i in range(30, 180, grid_size):
                cv2.line(heatmap, (0, i), (240, i), (30, 30, 30), 1)
            
            # Calculate activity density with higher resolution
            activity_grid = np.zeros((8, 12), dtype=float)  # Higher resolution grid
            
            for detection in detections_history[-100:]:  # Last 100 detections
                bbox = detection.get("bbox", [0, 0, 0, 0])
                if len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    center_x = int((x + w/2) / frame.shape[1] * 12)
                    center_y = int((y + h/2) / frame.shape[0] * 8)
                    
                    if 0 <= center_x < 12 and 0 <= center_y < 8:
                        # Add weighted contribution based on threat score
                        threat_score = detection.get("threat_score", 1.0)
                        activity_grid[center_y, center_x] += threat_score
            
            # Apply Gaussian smoothing for better visualization
            from scipy.ndimage import gaussian_filter
            try:
                activity_grid = gaussian_filter(activity_grid, sigma=1.0)
            except:
                pass  # Fallback if scipy not available
            
            # Draw enhanced heatmap with gradient
            max_activity = np.max(activity_grid) if np.max(activity_grid) > 0 else 1
            
            for y in range(8):
                for x in range(12):
                    intensity = activity_grid[y, x] / max_activity
                    
                    # Enhanced color gradient
                    if intensity < 0.2:
                        # Dark blue to blue
                        color = (0, 0, int(intensity * 5 * 255))
                    elif intensity < 0.4:
                        # Blue to cyan
                        ratio = (intensity - 0.2) / 0.2
                        color = (0, int(ratio * 255), 255)
                    elif intensity < 0.6:
                        # Cyan to green
                        ratio = (intensity - 0.4) / 0.2
                        color = (0, 255, int((1 - ratio) * 255))
                    elif intensity < 0.8:
                        # Green to yellow
                        ratio = (intensity - 0.6) / 0.2
                        color = (int(ratio * 255), 255, 0)
                    else:
                        # Yellow to red
                        ratio = (intensity - 0.8) / 0.2
                        color = (255, int((1 - ratio) * 255), 0)
                    
                    # Draw cell with gradient effect
                    cell_x = x * grid_size
                    cell_y = 30 + y * grid_size
                    
                    # Main cell
                    cv2.rectangle(heatmap, (cell_x, cell_y), 
                                 (cell_x + grid_size, cell_y + grid_size), color, -1)
                    
                    # Add subtle border
                    cv2.rectangle(heatmap, (cell_x, cell_y), 
                                 (cell_x + grid_size, cell_y + grid_size), 
                                 (color[0]//2, color[1]//2, color[2]//2), 1)
                    
                    # Add activity count for high activity areas
                    if activity_grid[y, x] > 0.5:
                        count = int(activity_grid[y, x])
                        cv2.putText(heatmap, str(count), 
                                   (cell_x + 5, cell_y + 12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add intensity scale
            scale_x = 220
            scale_height = 120
            for i in range(scale_height):
                intensity = i / scale_height
                if intensity < 0.2:
                    color = (0, 0, int(intensity * 5 * 255))
                elif intensity < 0.4:
                    ratio = (intensity - 0.2) / 0.2
                    color = (0, int(ratio * 255), 255)
                elif intensity < 0.6:
                    ratio = (intensity - 0.4) / 0.2
                    color = (0, 255, int((1 - ratio) * 255))
                elif intensity < 0.8:
                    ratio = (intensity - 0.6) / 0.2
                    color = (int(ratio * 255), 255, 0)
                else:
                    ratio = (intensity - 0.8) / 0.2
                    color = (255, int((1 - ratio) * 255), 0)
                
                cv2.line(heatmap, (scale_x, 30 + scale_height - i), 
                        (scale_x + 10, 30 + scale_height - i), color, 2)
            
            # Scale labels
            cv2.putText(heatmap, "HIGH", (scale_x - 5, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            cv2.putText(heatmap, "LOW", (scale_x - 5, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        return heatmap
    
    def create_vertical_analytics_panel(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None) -> np.ndarray:
        """Create vertical analytics panel for right side"""
        analytics = np.zeros((540, 240, 3), dtype=np.uint8)
        analytics.fill(20)  # Dark background
        
        # Add title
        cv2.putText(analytics, "SYSTEM ANALYTICS", (50, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Get current system state
        current_state = getattr(self.decision_engine.state_agent, 'state_transition', None)
        system_state = current_state.current_state.value if current_state else "UNKNOWN"
        
        # Analytics data
        analytics_data = [
            ("System State:", system_state.upper(), self.get_system_state_color(system_state.upper())),
            ("", "", (255, 255, 255)),
            ("📊 Statistics", "", (0, 255, 0)),
            ("Total:", str(self.stats['total_detections']), (255, 255, 255)),
            ("Threat:", str(self.stats['threat_detections']), (255, 165, 0)),
            ("Threat Rate:", f"{(self.stats['threat_detections']/max(1,self.stats['total_detections'])*100):.1f}%", (255, 165, 0)),
            ("Evidence:", str(len([f for f in os.listdir("evidence/videos") if f.endswith(".mp4")])), (0, 255, 255)),
            ("", "", (255, 255, 255)),
            ("⚡ Performance", "", (0, 255, 0)),
            ("FPS:", "30.0", (0, 255, 0)),
            ("CPU:", "45%", (255, 255, 0)),
            ("Memory:", "512MB", (255, 165, 0)),
            ("", "", (255, 255, 255)),
            ("⏱️ Timeline", "", (0, 255, 255)),
            ("Uptime:", "02:15:30", (0, 255, 255)),
            ("Last Alert:", "None", (255, 255, 255)),
            ("", "", (255, 255, 255)),
            ("🔥 Activity", "", (255, 165, 0)),
            ("Last Hour:", str(len([d for d in getattr(self, 'detection_history', [])[-50:]])), (255, 165, 0)),
            ("Peak:", "12/min", (255, 0, 0)),
        ]
        
        y_offset = 50
        for label, value, color in analytics_data:
            if label:
                if label.startswith(("📊", "⚡", "⏱️", "🔥")):
                    # Category headers
                    cv2.putText(analytics, label, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    # Regular labels and values
                    cv2.putText(analytics, label, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    cv2.putText(analytics, value, (120, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_offset += 18
        
        # Add weapon details section
        cv2.putText(analytics, "WEAPONS DETECTED", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += 20
        
        if detections is None:
            detections = []
        
        if detections:
            for i, detection in enumerate(detections[:5]):  # Show max 5 weapons
                weapon_type = detection.get("name", "Unknown").upper()
                weapon_id = detection.get("id", i+1)
                confidence = detection.get("confidence", 0) * 100
                
                # Color code by weapon type
                if "GUN" in weapon_type or "PISTOL" in weapon_type or "RIFLE" in weapon_type:
                    color = (0, 0, 255)  # Red for guns
                elif "KNIFE" in weapon_type or "BLADE" in weapon_type:
                    color = (0, 255, 255)  # Yellow for knives
                else:
                    color = (255, 165, 0)  # Orange for other weapons
                
                cv2.putText(analytics, f"ID:{weapon_id} {weapon_type}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                cv2.putText(analytics, f"{confidence:.1f}%", (150, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y_offset += 15
        else:
            cv2.putText(analytics, "No weapons detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
            y_offset += 15
        
        return analytics
    
    def create_four_section_display(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                                   results: List[Dict[str, Any]]) -> np.ndarray:
        """Create 4-section display layout with larger main feed"""
        # Get original frame dimensions
        height, width = frame.shape[:2]
        
        # Create full screen canvas (1280x720)
        full_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        full_screen.fill(10)  # Dark background
        
        # Section 1: Original Feed (Left Side - 800x600) - Larger
        original_section = cv2.resize(frame, (800, 600))
        full_screen[60:660, 0:800] = original_section
        
        # Section 2: Bird's Eye View (Top Right - 240x180) - Smaller
        birds_eye = self.create_birds_eye_view(frame, detections)
        birds_eye_small = cv2.resize(birds_eye, (240, 180))
        full_screen[60:240, 800:1040] = birds_eye_small
        
        # Section 3: Enhanced Heatmap (Bottom Right - 240x180) - Smaller but Enhanced
        # Store detection history for heatmap
        if not hasattr(self, 'detection_history'):
            self.detection_history = []
        self.detection_history.extend(detections)
        self.detection_history = self.detection_history[-100:]  # Keep last 100
        
        heatmap = self.create_enhanced_heatmap(frame, self.detection_history)
        heatmap_small = cv2.resize(heatmap, (240, 180))
        full_screen[240:420, 800:1040] = heatmap_small
        
        # Section 4: System Analytics (Right Side - 240x540) - Vertical Panel
        analytics = self.create_vertical_analytics_panel(frame, detections)
        full_screen[60:600, 1040:1280] = analytics
        
        # Add section borders
        cv2.rectangle(full_screen, (0, 60), (800, 660), (0, 255, 0), 2)
        cv2.rectangle(full_screen, (800, 60), (1040, 240), (0, 255, 0), 2)
        cv2.rectangle(full_screen, (800, 240), (1040, 420), (0, 255, 0), 2)
        cv2.rectangle(full_screen, (1040, 60), (1280, 600), (0, 255, 0), 2)
        
        # Add section labels
        cv2.putText(full_screen, "LIVE FEED", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(full_screen, "BIRD'S EYE", (810, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(full_screen, "HEATMAP", (860, 265), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(full_screen, "ANALYTICS", (1100, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add main header
        cv2.rectangle(full_screen, (0, 0), (1280, 60), (20, 20, 20), -1)
        cv2.putText(full_screen, "🎯 INTELLIGENT WEAPON DETECTION SYSTEM | AI-Powered Security Monitoring", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(full_screen, timestamp, (1050, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Add bottom status bar
        cv2.rectangle(full_screen, (0, 660), (1280, 720), (20, 20, 20), -1)
        cv2.rectangle(full_screen, (0, 660), (1280, 720), (0, 255, 0), 1)
        
        # Status info
        current_state = getattr(self.decision_engine.state_agent, 'state_transition', None)
        system_state = current_state.current_state.value if current_state else "UNKNOWN"
        state_color = self.get_system_state_color(system_state.upper())
        
        status_text = f"State: {system_state.upper()} | Total: {self.stats['total_detections']} | Threat: {self.stats['threat_detections']} | FPS: 30.0"
        cv2.putText(full_screen, status_text, (20, 685), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        controls = "[Q] Quit [S] Save [R] Reset [E] Evidence"
        cv2.putText(full_screen, controls, (20, 705), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return full_screen
    
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
                
                # Create 4-section display
                display_frame = self.create_four_section_display(frame, detections, results)
                
                # Display frame with professional window title
                cv2.imshow("🎯 Intelligent Weapon Detection System | AI-Powered Security", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_manual_frame(display_frame)  # Save full 4-section display
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
