"""
Real-Time Weapon Detection System
Professional Entry Point

This is the main entry point for the Intelligent Weapon Detection System.
It combines YOLO-based object detection with multi-agent decision making
for real-time threat assessment and response.

Author: AI Security Systems
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.integrated_gun_detection_system import IntegratedGunDetectionSystem

def main():
    """Main entry point for the weapon detection system"""
    print("=" * 80)
    print("🎯 INTELLIGENT WEAPON DETECTION SYSTEM")
    print("   AI-Powered Real-Time Security Monitoring")
    print("=" * 80)
    
    # Check if model exists
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} model not found!")
        print(f"Please ensure {model_path} is available")
        return
    
    try:
        # Initialize and run system
        print("🚀 Initializing system components...")
        system = IntegratedGunDetectionSystem(model_path=model_path)
        print("✅ System ready - Starting detection...")
        system.run()
        
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
