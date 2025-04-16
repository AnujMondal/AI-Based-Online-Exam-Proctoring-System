# AI-Based Online Examination Proctoring System

A comprehensive web-based system that uses artificial intelligence to proctor online examinations, ensuring academic integrity through advanced monitoring and detection features.

## Features

- **Face Detection & Identity Verification**: Ensures the correct student is taking the test
- **Multiple Face Detection**: Alerts when more than one person is detected in the camera feed
- **Head Pose Estimation**: Monitors suspicious head movements
- **Object Detection**: Identifies prohibited items like phones, books, or notes
- **Audio Monitoring**: Detects multiple voices or suspicious sounds
- **Browser Activity Monitoring**: Prevents tab switching or using keyboard shortcuts
- **Real-time Alerts**: Immediate notifications for suspicious activities
- **Proctor Dashboard**: Administrative interface for monitoring multiple test sessions
- **Session Management**: Track exam progress and view violation reports

## System Architecture

```
enhanced_proctoring/
├── app.py                    # Flask application
├── requirements.txt          # Dependencies
├── face_detection.py         # Face detection module
├── identity_verification.py  # Identity verification module
├── head_pose_detection.py    # Head pose estimation module
├── object_detection.py       # Prohibited item detection
├── audio_monitoring.py       # Audio analysis module
├── session_manager.py        # Session management
├── static/                   # Static assets
│   └── uploads/              # User uploads (photos, violations)
└── templates/                # HTML templates
    ├── index.html            # Landing page
    ├── login.html            # Login page
    ├── setup.html            # Identity verification setup
    ├── test.html             # Exam interface with monitoring
    └── proctor_dashboard.html # Admin monitoring interface
```

## Setup Instructions

1. Create and activate a virtual environment:

   ```bash
   python -m venv exam_proctor_env
   source exam_proctor_env/bin/activate  # On Windows: exam_proctor_env\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   cd enhanced_proctoring
   pip install -r requirements.txt
   ```

3. Prepare directories:

   ```bash
   mkdir -p static/uploads/baseline
   mkdir -p static/uploads/violations
   mkdir -p static/uploads/temp
   mkdir -p sessions
   ```

4. Run the application:

   ```bash
   python app.py
   ```

5. Access the application:
   Open your browser and navigate to `http://127.0.0.1:5000`

## Usage Flow

1. **Student Login**: Students log in with their credentials
2. **Identity Setup**: Students provide a photo and voice sample for verification
3. **Exam Session**: The system monitors the student during the exam
4. **Real-time Monitoring**: Violations are tracked and warnings are issued
5. **Exam Submission**: Results are saved along with proctoring data

## Development Notes

- The system requires camera and microphone access
- For face detection to work properly, ensure good lighting conditions
- Head pose detection works best when the face is clearly visible
- Object detection is optimized for common prohibited items

## License

This project is licensed under the MIT License - see the LICENSE file for details.
