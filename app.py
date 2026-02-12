"""
AI Yoga Instructor MVP - Main Application
Main Flask application that coordinates all components
"""
from __future__ import annotations
import cv2
import threading
import time
import pymysql
from datetime import datetime
from collections import deque, Counter
from queue import Queue, Empty
from functools import wraps
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_bcrypt import Bcrypt
from flask_login import (
    LoginManager, UserMixin, current_user,
    login_required, login_user, logout_user
)

# Import our custom modules
from pose_detection import PoseDetector
from pose_classifier import PoseClassifier
from progress_tracker import ProgressTracker
from voice_feedback import VoiceFeedback
from config import get_config

app = Flask(__name__)
conf = get_config()
app.config["SECRET_KEY"] = conf.SECRET_KEY

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


# ---------------- PER USER SESSION STATE ---------------- #

class SessionState:
    def __init__(self, user_id):
        self.user_id = user_id
        self.active = False
        self.camera = None

        # --- PRODUCTION FIX: AI Models MUST be per-session ---
        # We start them as None and load them only when streaming starts.
        self.detector = None
        self.classifier = None

        self.current_pose = None
        self.pose_start_time = None
        self.pose_duration = 0.0
        self.last_accuracy = 0.0

        self.pose_buffer = deque(maxlen=8)
        self.is_processing = False
        self.lock = threading.Lock()
        
        self.log_queue = Queue()
        self.log_thread = None

    def start_ai_engine(self):
        """Lazy load AI models to ensure thread safety"""
        if not self.detector:
            conf = get_config()
            self.detector = PoseDetector(
                model_complexity=conf.POSE_MODEL_COMPLEXITY,
                min_detection_confidence=conf.POSE_DETECTION_CONFIDENCE,
                min_tracking_confidence=conf.POSE_TRACKING_CONFIDENCE
            )
        if not self.classifier:
            self.classifier = PoseClassifier()

    def cleanup(self):
        """Aggressively free resources"""
        self.active = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        if self.detector:
            self.detector.cleanup()
            self.detector = None
        self.classifier = None
        
        if self.log_queue:
            self.log_queue.put(None)

user_states = {}

def get_user_state(user_id):
    if user_id not in user_states:
        user_states[user_id] = SessionState(user_id)
    return user_states[user_id]


# ---------------- SYSTEM COMPONENTS ---------------- #

progress_tracker = None
voice_feedback = None


class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email


@login_manager.user_loader
def load_user(user_id):
    conn = None
    try:
        conn = pymysql.connect(**get_config().DB_CONFIG)
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        user = cur.fetchone()
        if user:
            return User(user["id"], user["username"], user["email"])
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        if conn: conn.close()
    return None


def initialize_components():
    """Initialize shared system components"""
    global progress_tracker, voice_feedback
    try:
        conf = get_config()
        progress_tracker = ProgressTracker(conf.DB_CONFIG)
        voice_feedback = VoiceFeedback()
        print("Shared components initialized successfully")
    except Exception as e:
        print(f"Error initializing components: {e}")


# ---------------- UTILITIES ---------------- #

REQUEST_LOG = {}
RATE_LIMIT = 60
WINDOW = 60

def rate_limit(key):
    now = time.time()
    if key not in REQUEST_LOG:
        REQUEST_LOG[key] = []
    
    # PRODUCTION FIX: Filter list to remove old timestamps (Memory Leak Fix)
    REQUEST_LOG[key] = [t for t in REQUEST_LOG[key] if now - t < WINDOW]

    if len(REQUEST_LOG[key]) >= RATE_LIMIT:
        return False

    REQUEST_LOG[key].append(now)
    return True


def log_worker(state: SessionState):
    """Background DB logger for a user session."""
    while True:
        try:
            try:
                job = state.log_queue.get(timeout=2)
            except Empty:
                if not state.active: 
                    break
                continue

            if job is None:
                break

            if progress_tracker:
                progress_tracker.log_pose(**job)
            
            state.log_queue.task_done()
        except Exception:
            continue


def api_response(ok=True, data=None, message="", status=200):
    return jsonify({
        "ok": ok,
        "data": data,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }), status


# ---------------- VIDEO PROCESSING ---------------- #

CONF_THRESHOLD = 60.0 
FRAME_DELAY = 0.03  # ~30 FPS

def generate_frames(user_id):
    """Generate video frames with pose detection overlay"""
    state = get_user_state(user_id)

    if state.camera is None:
        state.camera = cv2.VideoCapture(0)
        if not state.camera.isOpened():
            print("Error: Could not open camera")
            return

    # Load AI for THIS session/thread
    state.start_ai_engine()

    try:
        while state.active:
            success, frame = state.camera.read()
            if not success:
                break

            # Use LOCAL detector instances
            if state.detector and state.classifier:
                results = state.detector.detect_pose(frame)

                if results.pose_landmarks:
                    # FIX: Use normalized landmarks for classification
                    norm_landmarks = state.detector.get_normalized_landmarks(results)
                    pose_info = state.classifier.classify_pose(norm_landmarks)
                    
                    raw_pose = pose_info.get("pose_name", "Unknown")
                    accuracy = pose_info.get("accuracy", 0.0)

                    # Smoothing
                    if accuracy >= CONF_THRESHOLD:
                        state.pose_buffer.append(raw_pose)
                    else:
                        state.pose_buffer.append("Unknown")

                    if state.pose_buffer:
                        smoothed_pose = Counter(state.pose_buffer).most_common(1)[0][0]
                    else:
                        smoothed_pose = None

                    if smoothed_pose == "Unknown":
                        smoothed_pose = None

                    with state.lock:
                        if smoothed_pose and smoothed_pose != state.current_pose:
                            if state.current_pose and state.pose_start_time:
                                duration = time.time() - state.pose_start_time
                                
                                # Queue DB Log
                                state.log_queue.put({
                                    "user_id": state.user_id,
                                    "pose_name": state.current_pose,
                                    "duration_held": duration,
                                    "accuracy_score": state.last_accuracy
                                })

                                if voice_feedback:
                                    voice_feedback.speak(f"Great job! You held {state.current_pose} for {duration:.1f} seconds")

                            state.current_pose = smoothed_pose
                            state.pose_start_time = time.time()
                            state.pose_duration = 0
                            state.last_accuracy = accuracy

                    # Draw Overlay
                    frame = state.detector.draw_landmarks(frame, results)
                    frame = state.classifier.draw_pose_info(frame, pose_info)

                    if state.pose_start_time:
                        state.pose_duration = time.time() - state.pose_start_time
                        cv2.putText(frame, f"Hold: {state.pose_duration:.1f}s", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(FRAME_DELAY)

    except GeneratorExit:
        pass
    finally:
        if state.camera:
            state.camera.release()
            state.camera = None

# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    if not rate_limit(request.remote_addr):
        return api_response(False, message="Too many requests", status=429)
    return render_template('index.html', logged_in=current_user.is_authenticated)

@app.route('/video_feed')
@login_required
def video_feed():
    if not rate_limit(request.remote_addr):
        return api_response(False, message="Too many requests", status=429)

    state = get_user_state(current_user.id)
    if not state.active:
        return "", 204
    
    return Response(generate_frames(current_user.id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/pose_info')
@login_required
def get_pose_info():
    state = get_user_state(current_user.id)
    return api_response(True, {
        "current_pose": state.current_pose,
        "duration": round(state.pose_duration, 1)
    })

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    data = request.get_json()
    if len(data.get("password", "")) < 4:
        return jsonify({"error": "Password too short"}), 400
    
    password_hash = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    conn = None
    try:
        conn = pymysql.connect(**get_config().DB_CONFIG)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, email, password_hash, age, height, weight) VALUES (%s,%s,%s,%s,%s,%s)", 
                   (data["username"], data["email"], password_hash, data.get("age"), data.get("height"), data.get("weight")))
        conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        if conn: conn.close()

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    data = request.get_json()
    conn = None
    try:
        conn = pymysql.connect(**get_config().DB_CONFIG)
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("SELECT * FROM users WHERE username=%s", (data["username"],))
        user = cur.fetchone()
        if user and bcrypt.check_password_hash(user["password_hash"], data["password"]):
            login_user(User(user["id"], user["username"], user["email"]))
            return jsonify({"status": "success"})
        return jsonify({"error": "Invalid credentials"}), 401
    finally:
        if conn: conn.close()

@app.route('/api/start_session')
@login_required
def start_session():
    state = get_user_state(current_user.id)
    if not state.is_processing:
        if progress_tracker:
            session_id = progress_tracker.start_session(state.user_id)
        else:
            session_id = 1
        state.is_processing = True
        state.active = True
        
        if not state.log_thread or not state.log_thread.is_alive():
            state.log_thread = threading.Thread(target=log_worker, args=(state,), daemon=True)
            state.log_thread.start()
            
        if voice_feedback:
            voice_feedback.speak("Welcome! Let's begin.")
        return api_response(True, {"session_id": session_id}, "Session started")
    return api_response(False, message="Session already active", status=400)

@app.route('/api/end_session')
@login_required
def end_session():
    state = get_user_state(current_user.id)
    if state.active:
        if state.current_pose and state.pose_start_time:
            duration = time.time() - state.pose_start_time
            if duration > 0:
                state.log_queue.put({"user_id": state.user_id, "pose_name": state.current_pose, "duration_held": duration, "accuracy_score": state.last_accuracy})
        
        if progress_tracker:
            progress_tracker.close_session(state.user_id)
        state.cleanup()
        if voice_feedback:
            voice_feedback.speak("Session ended.")
        return api_response(True, message="Session ended")
    return api_response(False, message="No active session", status=400)

@app.route("/logout")
@login_required
def logout():
    state = user_states.pop(current_user.id, None)
    if state: state.cleanup()
    logout_user()
    return redirect(url_for('index'))

@app.route('/api/progress')
@login_required
def get_progress():
    data = {}
    if progress_tracker:
        data = progress_tracker.get_user_progress(current_user.id) or {}
    return api_response(True, data)

if __name__ == '__main__':
    initialize_components()
    app.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False, threaded=True)