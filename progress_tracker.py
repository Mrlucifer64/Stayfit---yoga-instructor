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


    def update_user_progress_summary(self, user_id):
        """Aggregates all-time stats into the user_progress table for today"""
        conn = self._get_connection()
        if not conn: return
        
        try:
            with conn.cursor() as cursor:
                # Calculate daily stats
                cursor.execute("""
                    SELECT 
                        SUM(total_duration_seconds) as total_time,
                        SUM(poses_completed) as total_poses,
                        AVG(average_accuracy) as daily_avg
                    FROM yoga_sessions
                    WHERE user_id = %s AND DATE(session_start) = CURRENT_DATE
                """, (user_id,))
                
                stats = cursor.fetchone()
                if not stats or stats[0] is None: return

                total_time = stats[0]
                total_poses = stats[1]
                daily_accuracy = stats[2]
                
                # Insert or Update (Upsert) for today
                cursor.execute("""
                    INSERT INTO user_progress 
                    (user_id, log_date, total_practice_time_seconds, poses_practiced, improvement_score)
                    VALUES (%s, CURRENT_DATE, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    total_practice_time_seconds = VALUES(total_practice_time_seconds),
                    poses_practiced = VALUES(poses_practiced),
                    improvement_score = VALUES(improvement_score)
                """, (user_id, total_time, total_poses, daily_accuracy))
                
        except Exception as e:
            print(f"Error updating user progress: {e}")
        finally:
            conn.close()
    
    # ---------------- CLOSE SESSION ---------------- #

    def close_session(self, user_id):
        """End the session and calculate summary stats from the logs"""
        conn = self._get_connection()
        if not conn: return
        
        try:
            with conn.cursor() as cursor:
                # 1. Get the current active session ID
                cursor.execute("""
                    SELECT id, session_start FROM yoga_sessions 
                    WHERE user_id = %s AND session_end IS NULL 
                    ORDER BY session_start DESC LIMIT 1
                """, (user_id,))
                
                session = cursor.fetchone()
                if not session: return # No active session to close
                
                session_id = session[0]
                start_time = session[1]
                
                # 2. Calculate Stats from Pose Logs
                cursor.execute("""
                    SELECT 
                        COUNT(*) as pose_count, 
                        IFNULL(AVG(accuracy_score), 0) as avg_acc 
                    FROM pose_logs 
                    WHERE session_id = %s
                """, (session_id,))
                
                stats = cursor.fetchone()
                poses_completed = stats[0]
                avg_accuracy = stats[1]
                
                # 3. Calculate Duration (Now - Start Time)
                # We do this in Python to be precise, or let MySQL do it with TIMESTAMPDIFF
                
                # 4. Update the Session Record
                cursor.execute("""
                    UPDATE yoga_sessions 
                    SET 
                        session_end = CURRENT_TIMESTAMP,
                        total_duration_seconds = TIMESTAMPDIFF(SECOND, session_start, CURRENT_TIMESTAMP),
                        poses_completed = %s,
                        average_accuracy = %s
                    WHERE id = %s
                """, (poses_completed, avg_accuracy, session_id))
                
                # 5. Trigger User Progress Update (See Step 3 below)
                self.update_user_progress_summary(user_id)
                
        except Exception as e:
            print(f"Error closing session: {e}")
        finally:
            conn.close()