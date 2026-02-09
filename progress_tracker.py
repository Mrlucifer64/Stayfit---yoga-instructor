import os
import pymysql
from pymysql import Error
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
from config import get_config
from pose_detection import PoseDetector
from pose_classifier import PoseClassifier
from voice_feedback import VoiceFeedback


class ProgressTracker:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
        self._initialize_database()
        
    def _initialize_database(self):
        try:
            self.connection = pymysql.connect(**self.db_config)  
            print("MySQL connection established successfully")
        except Error as e:
            print(f"MySQL connection failed: {e}")
            self.connection = None
            self._use_fallback_storage()
            


    def start_session(self) -> int:
        if self.connection:
            cursor = self.connection.cursor()
            cursor.execute("INSERT INTO yoga_sessions (session_start) VALUES (NOW())")
            session_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()
            return session_id
        return 1 # Fallback ID

    def log_pose(self, pose_name: str, duration_held: float, 
                 accuracy_score: float, pose_metrics: Optional[Dict] = None,
                 feedback_notes: Optional[str] = None, session_id: Optional[int] = None):
        if not session_id:
            session_id = self._get_current_session_id()
        
        if self.connection :
            cursor = self.connection.cursor()
            query = """
                INSERT INTO pose_logs 
                (session_id, pose_name, duration_held_seconds, accuracy_score, pose_metrics, feedback_notes)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (session_id, pose_name, int(duration_held), 
                                  accuracy_score, json.dumps(pose_metrics), feedback_notes))
            self.connection.commit()
            cursor.close()

    def _get_current_session_id(self) -> int:
        if self.connection :
            cursor = self.connection.cursor()
            cursor.execute("SELECT id FROM yoga_sessions WHERE session_end IS NULL ORDER BY session_start DESC LIMIT 1")
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else self.start_session()
        return 1

    
            
    def _use_fallback_storage(self):
     print("Warning: Database unavailable. Using local fallback (No-op).")
     self.connection = None
    
    def close_session(self, session_id: Optional[int] = None):
        try:
            if self.connection :
                if not session_id:
                    session_id = self._get_current_session_id()
                cursor = self.connection.cursor()
                cursor.execute("UPDATE yoga_sessions SET session_end = NOW() WHERE id = %s", (session_id,))
                self.connection.commit()
                cursor.close()
        except Exception as e:
            print(f"Error closing session: {e}")
     
def initialize_components():
    global pose_detector, pose_classifier, progress_tracker, voice_feedback

    try:
        conf = get_config()  # Load DB config
        pose_detector = PoseDetector()
        pose_classifier = PoseClassifier()
        progress_tracker = ProgressTracker(conf.DB_CONFIG) 
        voice_feedback = VoiceFeedback()
        print("All components initialized successfully")
    except Exception as e:
        print(f"Error initializing components: {e}")