"""
AI Yoga Instructor - Final Production Version
"""
from __future__ import annotations
import cv2
import threading
import time
import pymysql
from datetime import datetime
from collections import deque, Counter
from queue import Queue, Empty
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user

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

class SessionState:
    def __init__(self, user_id):
        self.user_id = user_id
        self.active = False
        self.camera = None
        self.detector = None
        self.classifier = None
        self.voice_feedback = VoiceFeedback()
        self.current_pose = None
        self.previous_pose = None
        self.pose_start_time = None
        self.last_accuracy = 0.0
        self.pose_buffer = deque(maxlen=8)
        self.is_processing = False
        self.lock = threading.Lock()
        self.log_queue = Queue()
        self.log_thread = None

    def start_ai_engine(self):
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
        self.active = False
        self.is_processing = False
        self.current_pose = None
        # Clear voice queue
        while not self.voice_feedback.message_queue.empty():
            self.voice_feedback.message_queue.get()
        if self.camera:
            self.camera.release()
            self.camera = None
        if self.detector:
            self.detector.cleanup()
            self.detector = None
        self.classifier = None
        if self.log_queue:
            self.log_queue.put(None)
            self.log_queue = Queue()

user_states = {}

def get_user_state(user_id):
    if user_id not in user_states:
        user_states[user_id] = SessionState(user_id)
    return user_states[user_id]

progress_tracker = None

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
        if user: return User(user["id"], user["username"], user["email"])
    except Exception: pass
    finally:
        if conn: conn.close()
    return None

def initialize_components():
    global progress_tracker
    try:
        conf = get_config()
        progress_tracker = ProgressTracker(conf.DB_CONFIG)
        print("Components initialized successfully")
    except Exception as e: print(f"Init Error: {e}")

def api_response(ok=True, data=None, message="", status=200):
    return jsonify({"ok": ok, "data": data, "message": message}), status

def log_worker(state):
    while True:
        try:
            job = state.log_queue.get(timeout=2)
            if job is None: break
            if progress_tracker: progress_tracker.log_pose(**job)
            state.log_queue.task_done()
        except Empty:
            if not state.active: break

CONF_THRESHOLD = 60.0 

def generate_frames(user_id):
    state = get_user_state(user_id)
    if not state.camera: state.camera = cv2.VideoCapture(0)
    state.start_ai_engine()

    while state.active:
        success, frame = state.camera.read()
        if not success: break

        if state.detector and state.classifier:
            results = state.detector.detect_pose(frame)
            if results.pose_landmarks:
                norm = state.detector.get_normalized_landmarks(results)
                info = state.classifier.classify_pose(norm)
                raw, acc, metrics = info.get("pose_name", "Unknown"), info.get("accuracy", 0.0), info.get("metrics", {})

                if acc >= CONF_THRESHOLD: state.pose_buffer.append(raw)
                else: state.pose_buffer.append("Unknown")

                smooth = Counter(state.pose_buffer).most_common(1)[0][0] if state.pose_buffer else None
                if smooth == "Unknown": smooth = None

                with state.lock:
                    if smooth and smooth != state.current_pose:
                        if state.current_pose and state.pose_start_time:
                            dur = time.time() - state.pose_start_time
                            state.log_queue.put({"user_id": state.user_id, "pose_name": state.current_pose, "duration_held": dur, "accuracy_score": state.last_accuracy,"pose_metrics": metrics,"feedback_notes": "Completed"})
                            state.voice_feedback.speak_pose_feedback(state.current_pose, state.last_accuracy, dur)

                        if state.current_pose:
                            state.voice_feedback.speak_pose_transition(state.current_pose, smooth)
                        else:
                            state.voice_feedback.speak_pose_instruction(smooth)

                        state.previous_pose = state.current_pose
                        state.current_pose = smooth
                        state.pose_start_time = time.time()
                        state.last_accuracy = acc

                    elif state.current_pose:
                        state.last_accuracy = acc
                        if 40.0 < acc < 85.0:
                            fix = state.classifier.get_correction_feedback(state.current_pose, metrics)
                            if fix: state.voice_feedback.speak_correction(fix)
                        if acc > 60.0: state.voice_feedback.speak_motivation()

                frame = state.detector.draw_landmarks(frame, results)

        ret, buf = cv2.imencode('.jpg', frame)
        if ret: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)
    if state.camera: state.camera.release()

@app.route('/')
def index(): return render_template('index.html', logged_in=current_user.is_authenticated)

@app.route('/video_feed')
@login_required
def video_feed():
    state = get_user_state(current_user.id)
    if not state.active: return "", 204
    return Response(generate_frames(current_user.id), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/pose_info')
@login_required
def get_pose_info():
    state = get_user_state(current_user.id)
    voice_msg = state.voice_feedback.get_message()
    return api_response(True, {
        "current_pose": state.current_pose, 
        "duration": round(time.time() - state.pose_start_time, 1) if state.pose_start_time else 0,
        "voice": voice_msg
    })

@app.route('/api/start_session')
@login_required
def start_session():
    state = get_user_state(current_user.id)
    if not state.is_processing:
        if progress_tracker: progress_tracker.start_session(state.user_id)
        state.is_processing = True
        state.active = True
        state.pose_start_time = None
        state.log_thread = threading.Thread(target=log_worker, args=(state,), daemon=True)
        state.log_thread.start()
        state.voice_feedback.speak_session_start()
        return api_response(True, message="Session Started ")
    return api_response(False, message="Active", status=400)

@app.route('/api/end_session')
@login_required
def end_session():
    state = get_user_state(current_user.id)
    if state.active:
        if progress_tracker: progress_tracker.close_session(state.user_id)
        state.cleanup()
        return api_response(True, message="Session Ended")
    return api_response(False, message="No session", status=400)

# (Login/Register/Logout/Progress/Main - Standard)
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET": return render_template("register.html")
    data = request.get_json()
    pw = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    try:
        conn = pymysql.connect(**get_config().DB_CONFIG)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s,%s,%s)", (data["username"], data["email"], pw))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"error": str(e)}), 400

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET": return render_template("login.html")
    data = request.get_json()
    try:
        conn = pymysql.connect(**get_config().DB_CONFIG)
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("SELECT * FROM users WHERE username=%s", (data["username"],))
        user = cur.fetchone()
        conn.close()
        if user and bcrypt.check_password_hash(user["password_hash"], data["password"]):
            login_user(User(user["id"], user["username"], user["email"]))
            return jsonify({"status": "success"})
        return jsonify({"error": "Invalid"}), 401
    except Exception as e: return jsonify({"error": str(e)}), 400

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/api/progress')
@login_required
def get_progress():
    data = progress_tracker.get_user_progress(current_user.id) if progress_tracker else {}
    return api_response(True, data)

if __name__ == '__main__':
    initialize_components()
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)