# 🎯 Intelligent Weapon Detection System

An AI-powered real-time security monitoring system that combines YOLO-based object detection with multi-agent decision making for automated threat assessment and response.

## 🚀 Features

### Core Capabilities
- **Real-time Weapon Detection**: Advanced YOLO model for detecting guns and knives
- **Multi-Agent Decision Engine**: Intelligent threat assessment with automated response coordination
- **Color Live Feed**: Full-color camera feed with enhanced visualization
- **Professional 4-Section Display**: Live feed, bird's eye view, heatmap, and analytics panel
- **Evidence Recording**: Automatic video capture and storage of threat events
- **Alert System**: Multi-level threat alerts with audio notifications

### Weapon Classification
- **🔫 Guns/Firearms**: Red bounding boxes with firearm classification
- **🔪 Knives/Blades**: Yellow bounding boxes with blade weapon classification
- **⚠️ Other Weapons**: Orange bounding boxes for unknown weapon types

### Smart Features
- **Threat Scoring**: Advanced algorithm for threat level assessment
- **State Management**: Intelligent system states (normal, suspicious, emergency)
- **Evidence Buffering**: Pre/post event video recording
- **Performance Analytics**: Real-time statistics and system monitoring

## 📁 Project Structure

```
intelligent-weapon-detection/
├── 📂 core/                    # Core system files
│   ├── integrated_gun_detection_system.py
│   └── __init__.py
├── 📂 agents/                  # Multi-agent decision engine
│   ├── agent_based_decision_engine.py
│   └── __init__.py
├── 📂 detection/               # Object detection & tracking
│   ├── activity_detection.py
│   ├── human_tracker.py
│   └── __init__.py
├── 📂 models/                  # AI models
│   └── best.pt                 # Main detection model
├── 📂 config/                  # Configuration files
│   └── settings.py
├── 📂 evidence/                # Evidence storage
│   ├── videos/                 # Recorded events
│   └── images/                 # Snapshots
├── 📂 utils/                   # Utility functions
├── 📂 tests/                   # Test files
├── 📂 docs/                    # Documentation
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenCV 4.5+
- CUDA-compatible GPU (optional, for better performance)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd intelligent-weapon-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model**
   - Place `best.pt` in the `models/` directory
   - Ensure the model is trained for weapon detection

4. **Run the system**
   ```bash
   python main.py
   ```

## 🎮 Usage

### Controls
- **'q'**: Quit the application
- **'s'**: Save current frame snapshot
- **'r'**: Reset system statistics
- **'e'**: Open evidence folder
- **'w'**: Reset evidence recording session

### System Interface

The system displays a professional 4-section layout:

1. **📹 Live Feed** (Left): Main camera feed with colored bounding boxes
2. **🗺️ Bird's Eye View** (Top Right): Overhead perspective of detections
3. **🔥 Heatmap** (Bottom Right): Detection activity visualization
4. **📊 Analytics** (Right Side): Real-time statistics and weapon details

### Alert Levels

- **🟢 LOW**: Minor threat detected
- **🟡 MEDIUM**: Moderate threat level
- **🔴 HIGH**: Serious threat detected
- **🚨 CRITICAL**: Immediate danger

## ⚙️ Configuration

Key settings can be adjusted in `config/settings.py`:

```python
# Detection thresholds
MODEL_CONFIDENCE_THRESHOLD = 0.5
ALERT_THRESHOLD = 2.0

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Colors (BGR format)
COLORS = {
    'GUN': (0, 0, 255),      # Red
    'KNIFE': (0, 255, 255),  # Yellow
    'OTHER': (255, 165, 0)   # Orange
}
```

## 📊 Performance

- **Inference Speed**: 100-200ms per frame
- **Accuracy**: High precision for weapon detection
- **Real-time Processing**: 30 FPS capability
- **Memory Usage**: Optimized for continuous operation

## 🔧 Technical Architecture

### Core Components

1. **YOLO Detection Engine**: Fast and accurate object detection
2. **Multi-Agent System**: Intelligent decision-making agents
3. **Evidence Management**: Automated recording and storage
4. **Alert System**: Multi-level threat notifications
5. **Analytics Engine**: Real-time performance monitoring

### Data Flow

```
Camera Input → YOLO Detection → Agent Analysis → Threat Assessment → Response Action
     ↓              ↓                ↓               ↓                ↓
   Frame        Bounding Boxes    Decision Logic   Alert Level    Evidence Storage
```

## 🚨 Safety & Ethics

- **Privacy First**: No personal data stored long-term
- **Responsible AI**: Designed for security applications only
- **Human Oversight**: Requires human verification for critical actions
- **Compliance**: Follows security and privacy regulations

## 📝 Development

### Adding New Features

1. **New Detection Classes**: Update `WEAPON_CLASSES` in settings
2. **Custom Agents**: Extend the agent-based decision engine
3. **Alert Types**: Modify the alert system configuration
4. **UI Changes**: Update the display layout functions

### Testing

```bash
# Run tests (when implemented)
python -m pytest tests/

# Test individual components
python -m detection.activity_detection
python -m agents.agent_based_decision_engine
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

- **AI Security Systems** - Development Team
- **Version**: 1.0.0
- **Last Updated**: 2024

## 📞 Support

For technical support or questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation in the `docs/` folder

---

**⚠️ Disclaimer**: This system is designed for legitimate security applications only. Users must comply with all applicable laws and regulations regarding surveillance and privacy.
