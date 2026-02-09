"""
AI Yoga Instructor MVP - Main Application
Main Flask application that coordinates all components
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
from datetime import datetime



# Import our custom modules
from pose_detection import PoseDetector
from pose_classifier import PoseClassifier
from progress_tracker import ProgressTracker
from voice_feedback import VoiceFeedback
from config import get_config

app = Flask(__name__)

# Global variables for video processing
camera = None
pose_detector = None
pose_classifier = None
progress_tracker = None
voice_feedback = None
current_pose = None
pose_start_time = None
pose_duration = 0
is_processing = False

def initialize_components():
    """Initialize all system components"""
    global pose_detector, pose_classifier, progress_tracker, voice_feedback
    
    try:
        conf = get_config()
        pose_detector = PoseDetector()
        pose_classifier = PoseClassifier()
        progress_tracker = ProgressTracker(conf.DB_CONFIG)
        voice_feedback = VoiceFeedback()
        print("All components initialized successfully")
    except Exception as e:
        print(f"Error initializing components: {e}")

def generate_frames():
    """Generate video frames with pose detection overlay"""
    global camera, pose_detector, pose_classifier, current_pose, pose_start_time, pose_duration
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera")
            return

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame")
            break
        
        # Process frame for pose detection
        if pose_detector and pose_classifier:
            # Detect poses
            results = pose_detector.detect_pose(frame)
            
            if results.pose_landmarks:
                # Classify the pose
                pose_info = pose_classifier.classify_pose(results.pose_landmarks)
                
                # Update current pose tracking
                if pose_info['pose_name'] != current_pose:
                    if current_pose and pose_start_time:
                        # Log the previous pose
                        duration = time.time() - pose_start_time
                        progress_tracker.log_pose(current_pose, duration, pose_info['accuracy'])
                        
                        # Provide voice feedback
                        voice_feedback.speak(f"Great job! You held {current_pose} for {duration:.1f} seconds")
                    
                    current_pose = pose_info['pose_name']
                    pose_start_time = time.time()
                    pose_duration = 0
                
                # Draw pose landmarks and classification
                frame = pose_detector.draw_landmarks(frame, results)
                frame = pose_classifier.draw_pose_info(frame, pose_info)
                
                # Draw timer
                if pose_start_time:
                    pose_duration = time.time() - pose_start_time
                    cv2.putText(frame, f"Hold: {pose_duration:.1f}s", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            print("Frame encoded:", ret)

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page with video feed and controls"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/pose_info')
def get_pose_info():
    """API endpoint to get current pose information"""
    global current_pose, pose_duration
    
    return jsonify({
        'current_pose': current_pose,
        'duration': round(pose_duration, 1),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/start_session')
def start_session():
    """Start a new yoga session"""
    global is_processing, progress_tracker, voice_feedback
    
    if voice_feedback is None or progress_tracker is None:
        return jsonify({'status': 'error', 'message': 'Voice system not initialized'}), 500
    
    if not is_processing:
        try:
            session_id = progress_tracker.start_session()
            is_processing = True
            voice_feedback.speak("Welcome to your AI yoga session! Let's begin with some breathing exercises.")
            return jsonify({'status': 'success', 'message': 'Session started', 'session_id': session_id })
        except Exception as e:
            app.logger.error(f"Error starting session: {e}")
            return jsonify({'status': 'error', 'message': 'Failed to start session'}), 500
    print(f"Started session {session_id}")
    return jsonify({'status': 'error', 'message': 'Session already in progress'})

@app.route('/api/end_session')
def end_session():
    """End the current yoga session safely"""
    global is_processing, current_pose, pose_start_time, progress_tracker, voice_feedback

    if voice_feedback is None or progress_tracker is None:
        return jsonify({'status': 'error', 'message': 'System not fully initialized'}), 500

    if is_processing:
        try:
            # Log final pose if valid
            if current_pose and pose_start_time:
                duration = time.time() - pose_start_time
                if duration > 0:  # sanity check
                    progress_tracker.log_pose(current_pose, duration, 0.0)
                    session_id = progress_tracker.close_session()

            is_processing = False
            current_pose = None
            pose_start_time = None

            voice_feedback.speak("Excellent work! Your yoga session is complete. Namaste!")
            return jsonify({'status': 'success','message': 'Session ended'})
            
        except Exception as e:
            app.logger.error(f"Error ending session: {e}")
            print(f"Closed session {session_id}")
            return jsonify({'status': 'error', 'message': 'Failed to end session'}), 500
    return jsonify({'status': 'error', 'message': 'No active session'})


@app.route('/api/progress')
def get_progress():
    """Get user's yoga progress data"""
    try:
        progress_data = progress_tracker.get_user_progress()
        return jsonify({'status': 'success', 'data': progress_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("Initializing AI Yoga Instructor...")
    initialize_components()
    
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=8080 , reloader=False)
