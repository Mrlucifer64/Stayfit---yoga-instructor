"""
Pose Detection Module
Handles real-time pose detection using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List

class PoseDetector:
    """MediaPipe-based pose detector for yoga poses"""
    
    def __init__(self):
        """Initialize MediaPipe pose detection"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection with optimized parameters
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define key body parts for yoga pose analysis
        self.key_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_HEEL,
            self.mp_pose.PoseLandmark.RIGHT_HEEL,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        ]
    
    def detect_pose(self, frame):
        """Detect poses in the given frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        return results
    
    def draw_landmarks(self, frame, results):
        """Draw pose landmarks on the frame"""
        if results.pose_landmarks:
            # Draw the pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Draw additional connections for better visualization
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return frame
    
    def get_landmark_coordinates(self, results, landmark_id):
        """Get coordinates of a specific landmark"""
        if results.pose_landmarks:
            landmark = results.pose_landmarks.landmark[landmark_id]
            return (landmark.x, landmark.y, landmark.z)
        return None
    
    def calculate_angle(self, a: Tuple[float, float, float], 
                       b: Tuple[float, float, float], 
                       c: Tuple[float, float, float]) -> float:
        """Calculate angle between three points"""
        if not all([a, b, c]):
            return 0.0
        
        # Convert to numpy arrays for easier calculation
        a = np.array([a[0], a[1]])
        b = np.array([b[0], b[1]])
        c = np.array([c[0], c[1]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def get_pose_metrics(self, results):
        """Extract key metrics from pose detection results"""
        if not results.pose_landmarks:
            return None
        
        metrics = {}
        
        # Get shoulder positions
        left_shoulder = self.get_landmark_coordinates(results, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self.get_landmark_coordinates(results, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        
        # Get hip positions
        left_hip = self.get_landmark_coordinates(results, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = self.get_landmark_coordinates(results, self.mp_pose.PoseLandmark.RIGHT_HIP)
        
        # Get knee positions
        left_knee = self.get_landmark_coordinates(results, self.mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = self.get_landmark_coordinates(results, self.mp_pose.PoseLandmark.RIGHT_KNEE)
        
        # Get ankle positions
        left_ankle = self.get_landmark_coordinates(results, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = self.get_landmark_coordinates(results, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        # Calculate key angles
        if all([left_shoulder, left_hip, left_knee]):
            metrics['left_hip_angle'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
        
        if all([right_shoulder, right_hip, right_knee]):
            metrics['right_hip_angle'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
        
        if all([left_hip, left_knee, left_ankle]):
            metrics['left_knee_angle'] = self.calculate_angle(left_hip, left_knee, left_ankle)
        
        if all([right_hip, right_knee, right_ankle]):
            metrics['right_knee_angle'] = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # Calculate shoulder alignment
        if left_shoulder and right_shoulder:
            metrics['shoulder_alignment'] = abs(left_shoulder[1] - right_shoulder[1])
        
        # Calculate hip alignment
        if left_hip and right_hip:
            metrics['hip_alignment'] = abs(left_hip[1] - right_hip[1])
        
        return metrics
    
    def cleanup(self):
        """Clean up resources"""
        if self.pose:
            self.pose.close()
