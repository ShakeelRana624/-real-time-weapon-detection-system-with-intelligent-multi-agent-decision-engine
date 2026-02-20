"""
Person Memory Manager for Persistent ID Storage and Recovery
Handles JSON-based storage of person features and IDs
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class PersonMemoryManager:
    def __init__(self, memory_file: str = "data/person_memory.json"):
        """
        Initialize Person Memory Manager
        
        Args:
            memory_file: Path to JSON memory file
        """
        self.memory_file = memory_file
        self.person_memory = {}
        self.active_sessions = {}  # Current active tracking sessions
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        
        # Load existing memory
        self.load_memory()
        
        print(f"✓ Person Memory Manager initialized: {memory_file}")
    
    def load_memory(self):
        """Load person memory from JSON file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.person_memory = data.get('persons', {})
                    self.active_sessions = data.get('active_sessions', {})
                print(f"✓ Loaded {len(self.person_memory)} persons from memory")
        except Exception as e:
            print(f"⚠️ Error loading memory: {e}")
            self.person_memory = {}
            self.active_sessions = {}
    
    def save_memory(self):
        """Save person memory to JSON file"""
        try:
            data = {
                'persons': self.person_memory,
                'active_sessions': self.active_sessions,
                'last_updated': datetime.now().isoformat(),
                'total_persons': len(self.person_memory)
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Saved {len(self.person_memory)} persons to memory")
        except Exception as e:
            print(f"❌ Error saving memory: {e}")
    
    def add_or_update_person(self, person_id: int, features: np.ndarray, 
                           bbox: List[int], confidence: float, 
                           frame_number: int, camera_id: int = 0) -> bool:
        """
        Add or update person in memory
        
        Args:
            person_id: Unique person identifier
            features: Feature vector from person
            bbox: Bounding box coordinates
            confidence: Detection confidence
            frame_number: Current frame number
            camera_id: Camera identifier
            
        Returns:
            True if new person added, False if updated
        """
        current_time = time.time()
        
        if person_id not in self.person_memory:
            # New person
            self.person_memory[person_id] = {
                'id': person_id,
                'first_seen': current_time,
                'last_seen': current_time,
                'total_detections': 1,
                'features_history': [features.tolist()],
                'bbox_history': [bbox],
                'confidence_history': [confidence],
                'frame_history': [frame_number],
                'camera_history': [camera_id],
                'avg_confidence': confidence,
                'status': 'active',
                'session_start': current_time
            }
            print(f"➕ Added new person ID {person_id} to memory")
            return True
        else:
            # Update existing person
            person = self.person_memory[person_id]
            person['last_seen'] = current_time
            person['total_detections'] += 1
            
            # Update histories (keep last 10)
            person['features_history'].append(features.tolist())
            person['features_history'] = person['features_history'][-10:]
            
            person['bbox_history'].append(bbox)
            person['bbox_history'] = person['bbox_history'][-10:]
            
            person['confidence_history'].append(confidence)
            person['confidence_history'] = person['confidence_history'][-10:]
            
            person['frame_history'].append(frame_number)
            person['frame_history'] = person['frame_history'][-10:]
            
            person['camera_history'].append(camera_id)
            person['camera_history'] = person['camera_history'][-10:]
            
            # Update average confidence
            person['avg_confidence'] = sum(person['confidence_history']) / len(person['confidence_history'])
            
            person['status'] = 'active'
            
            print(f"🔄 Updated person ID {person_id} in memory")
            return False
    
    def get_person_features(self, person_id: int) -> Optional[np.ndarray]:
        """
        Get stored features for person ID
        
        Args:
            person_id: Person identifier
            
        Returns:
            Feature vector if found, None otherwise
        """
        if person_id in self.person_memory:
            features = self.person_memory[person_id]['features_history']
            if features:
                # Return average of last 3 feature vectors
                recent_features = features[-3:] if len(features) >= 3 else features
                avg_features = np.mean(recent_features, axis=0)
                return avg_features
        
        return None
    
    def find_matching_person(self, current_features: np.ndarray, 
                          bbox: List[int], confidence: float,
                          max_time_diff: float = 10.0) -> Optional[int]:
        """
        Find matching person from memory based on feature similarity
        
        Args:
            current_features: Current feature vector
            bbox: Current bounding box
            confidence: Current detection confidence
            max_time_diff: Maximum time difference in seconds
            
        Returns:
            Matching person ID if found, None otherwise
        """
        current_time = time.time()
        best_match = None
        best_score = 0.0
        
        for person_id, person_data in self.person_memory.items():
            # Check time difference
            time_diff = current_time - person_data['last_seen']
            
            if time_diff > max_time_diff:
                continue  # Too long ago, skip
            
            # Get stored features
            stored_features = self.get_person_features(person_id)
            
            if stored_features is not None:
                try:
                    # Calculate cosine similarity
                    from scipy.spatial.distance import cosine
                    similarity = 1 - cosine(current_features, stored_features)
                    
                    # Combined score with confidence and spatial proximity
                    center_x = bbox[0] + bbox[2] // 2
                    center_y = bbox[1] + bbox[3] // 2
                    
                    # Check last known position
                    if person_data['bbox_history']:
                        last_bbox = person_data['bbox_history'][-1]
                        last_center_x = last_bbox[0] + last_bbox[2] // 2
                        last_center_y = last_bbox[1] + last_bbox[3] // 2
                        
                        # Calculate distance
                        distance = ((center_x - last_center_x)**2 + (center_y - last_center_y)**2)**0.5
                        spatial_score = max(0, 1 - distance / 200)  # Normalize distance
                        
                        # Combined score
                        combined_score = 0.6 * similarity + 0.3 * confidence + 0.1 * spatial_score
                        
                        if combined_score > best_score and combined_score > 0.7:
                            best_score = combined_score
                            best_match = person_id
                            
                except Exception:
                    continue
        
        if best_match:
            print(f"🎯 Found matching person ID {best_match} (score: {best_score:.2f})")
            return best_match
        
        return None
    
    def mark_person_inactive(self, person_id: int, reason: str = "timeout"):
        """
        Mark person as inactive
        
        Args:
            person_id: Person identifier
            reason: Reason for inactivity
        """
        if person_id in self.person_memory:
            self.person_memory[person_id]['status'] = 'inactive'
            self.person_memory[person_id]['inactive_reason'] = reason
            self.person_memory[person_id]['inactive_time'] = time.time()
            print(f"⏸ Marked person ID {person_id} inactive: {reason}")
    
    def cleanup_old_memory(self, max_age_hours: float = 24.0):
        """
        Clean up old memory entries
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        persons_to_remove = []
        for person_id, person_data in self.person_memory.items():
            age = current_time - person_data['last_seen']
            
            if age > max_age_seconds:
                persons_to_remove.append(person_id)
        
        # Remove old persons
        for person_id in persons_to_remove:
            del self.person_memory[person_id]
            cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned {cleaned_count} old persons from memory")
            self.save_memory()  # Save after cleanup
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        active_count = sum(1 for p in self.person_memory.values() if p['status'] == 'active')
        inactive_count = sum(1 for p in self.person_memory.values() if p['status'] == 'inactive')
        
        return {
            'total_persons': len(self.person_memory),
            'active_persons': active_count,
            'inactive_persons': inactive_count,
            'memory_file': self.memory_file,
            'last_updated': self.person_memory.get('last_updated')
        }
    
    def force_save(self):
        """Force save memory to file"""
        self.save_memory()
        print("💾 Memory saved to file")
