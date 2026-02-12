import pymysql
from pymysql import Error
from datetime import datetime
from typing import Dict, Any, Optional
import json

class ProgressTracker:
    def __init__(self, db_config):
        self.db_config = db_config
        # We test connection on init, but don't hold it open
        self._test_connection()
        
    def _test_connection(self):
        try:
            conn = self._get_connection()
            if conn:
                print("MySQL connection check successful")
                conn.close()
        except Exception as e:
            print(f"MySQL connection check failed: {e}")

    def _get_connection(self):
        """Create a fresh connection for thread safety"""
        try:
            return pymysql.connect(**self.db_config, autocommit=True)
        except Error as e:
            print(f"DB Connect Error: {e}")
            return None

    # ---------------- SESSION MANAGEMENT ---------------- #

    def start_session(self, user_id: int) -> int:
        conn = self._get_connection()
        if not conn: return 1 # Fallback ID

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO yoga_sessions (user_id, session_start) VALUES (%s, NOW())",
                    (user_id,)
                )
                return cursor.lastrowid
        except Exception as e:
            print(f"Start Session Error: {e}")
            return 1
        finally:
            conn.close()

    # ---------------- POSE LOGGING ---------------- #

    def log_pose(self, user_id: int, pose_name: str, duration_held: float, 
                 accuracy_score: float, pose_metrics: Optional[Dict] = None, 
                 feedback_notes: Optional[str] = None, session_id: Optional[int] = None,
                 **kwargs): # Accept extra args to prevent crashes
        
        conn = self._get_connection()
        if not conn: return

        try:
            if not session_id:
                session_id = self._get_current_session_id(user_id)

            with conn.cursor() as cursor:
                query = """
                    INSERT INTO pose_logs
                    (user_id, session_id, pose_name, duration_held_seconds, accuracy_score, pose_metrics, feedback_notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    user_id, session_id, pose_name, int(duration_held),
                    accuracy_score, json.dumps(pose_metrics or {}), feedback_notes
                ))
        except Exception as e:
            print(f"Log Pose Error: {e}")
        finally:
            conn.close()

    # ---------------- HELPERS ---------------- #

    def get_user_progress(self, user_id: int) -> Dict[str, Any]:
        conn = self._get_connection()
        default_stats = {"sessions": 0, "accuracy": 0, "time_spent": 0}
        
        if not conn: return default_stats

        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        COUNT(DISTINCT s.id) AS sessions,
                        IFNULL(AVG(p.accuracy_score), 0) AS accuracy,
                        IFNULL(SUM(p.duration_held_seconds), 0) AS time_spent
                    FROM yoga_sessions s
                    LEFT JOIN pose_logs p ON s.id = p.session_id
                    WHERE s.user_id = %s
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "sessions": int(row["sessions"]),
                        "accuracy": round(float(row["accuracy"]), 2),
                        "time_spent": round(float(row["time_spent"]), 1),
                    }
                return default_stats
        except Exception as e:
            print(f"Get Progress Error: {e}")
            return default_stats
        finally:
            conn.close()
    
    def _get_current_session_id(self, user_id: int) -> int:
        conn = self._get_connection()
        if not conn: return 1

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id FROM yoga_sessions
                    WHERE user_id = %s AND session_end IS NULL
                    ORDER BY session_start DESC LIMIT 1
                """, (user_id,))
                result = cursor.fetchone()
                if result:
                    return result[0]
        except Exception:
            pass
        finally:
            conn.close()
            
        # If no active session found, try to make one (careful of recursion)
        return self.start_session(user_id)

    # ---------------- CLOSE SESSION ---------------- #

    def close_session(self, user_id: int, session_id: Optional[int] = None):
        conn = self._get_connection()
        if not conn: return

        try:
            with conn.cursor() as cursor:
                if not session_id:
                    # Logic inside _get_current_session_id handles connection itself
                    # so we don't call it here to avoid complexity, just update latest
                    cursor.execute("""
                        UPDATE yoga_sessions SET session_end = NOW()
                        WHERE user_id = %s AND session_end IS NULL
                    """, (user_id,))
                else:
                    cursor.execute("""
                        UPDATE yoga_sessions SET session_end = NOW()
                        WHERE id = %s AND user_id = %s
                    """, (session_id, user_id))
        except Exception as e:
            print(f"Close Session Error: {e}")
        finally:
            conn.close()