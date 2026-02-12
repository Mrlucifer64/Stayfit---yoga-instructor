"""
Pose Detection Module
Handles real-time pose detection using MediaPipe
"""
import math
import mediapipe as mp
import cv2
import numpy as np
from typing import Tuple, List, Optional

class PoseDetector:
    """MediaPipe-based pose detector for yoga poses"""
    
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe pose detection with configurable parameters
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
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
        if frame is None:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)
    
    def draw_landmarks(self, frame, results):
        """Draw pose landmarks on the frame"""
        if results and results.pose_landmarks:
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
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks to be translation and scale invariant. Returns list of (x, y, z)."""
        pts = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

        # --- Center on hips ---
        left_hip = pts[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = pts[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        center_x = (left_hip[0] + right_hip[0]) / 2
        center_y = (left_hip[1] + right_hip[1]) / 2
        center_z = (left_hip[2] + right_hip[2]) / 2

        centered = [
            (x - center_x, y - center_y, z - center_z)
            for x, y, z in pts
        ]

        # --- Scale using torso size ---
        left_shoulder = pts[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        
        # Calculate torso size using Euclidean distance
        torso_size = math.sqrt(
            (left_shoulder[0] - right_hip[0])**2 + 
            (left_shoulder[1] - right_hip[1])**2
        )

        # Prevent division by zero
        if torso_size < 1e-6:
            torso_size = 1.0

        # FIX: Critical indentation fix here. 
        # Previously, this block was inside the 'if' above, so it never ran for valid poses.
        normalized = [
            (x / torso_size, y / torso_size, z / torso_size)
            for x, y, z in centered
        ]
        return normalized

    def get_normalized_landmarks(self, results):
        """Return normalized landmarks from MediaPipe results."""
        if not results or not results.pose_landmarks:
            return None
        return self.normalize_landmarks(results.pose_landmarks)
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()