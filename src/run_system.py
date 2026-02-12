"""
Simple launcher for the Integrated Gun Detection System
"""

import sys
import os

def main():
    print(" INTEGRATED GUN DETECTION SYSTEM")
    print("=" * 50)
    
    # Check model
    import os
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best.pt")
    if not os.path.exists(model_path):
        print(f" Error: {model_path} not found!")
        print("Please ensure best.pt is in models/ directory")
        return
    
    # Check dependencies
    try:
        import cv2
        import ultralytics
        import numpy as np
        print("✓ Dependencies verified")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return
    
    # Import and run system
    try:
        from integrated_gun_detection_system import IntegratedGunDetectionSystem
        
        print("🚀 Starting detection system...")
        system = IntegratedGunDetectionSystem(model_path=model_path)
        system.run()
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
