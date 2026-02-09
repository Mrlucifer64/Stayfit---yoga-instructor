"""
Pose Classification Module
Classifies yoga poses based on MediaPipe pose landmarks
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple

class PoseClassifier:
    """Classifies yoga poses based on pose landmarks and geometric calculations"""
    
    def __init__(self):
        """Initialize the pose classifier with yoga pose definitions"""
        self.mp_pose = mp.solutions.pose
        
        # Define target angles and positions for each yoga pose
        self.pose_definitions = {
            'Tree Pose': {
                'hip_angle_range': (150, 180),
                'knee_angle_range': (150, 180),
                'foot_position': 'raised',
                'balance_required': True,
                'min_confidence': 0.7
            },
            'Warrior II': {
                'hip_angle_range': (80, 120),
                'knee_angle_range': (80, 120),
                'shoulder_alignment': 'horizontal',
                'arms_extended': True,
                'min_confidence': 0.7
            },
            'Downward Dog': {
                'hip_angle_range': (150, 180),
                'shoulder_hip_angle': (60, 120),
                'arms_straight': True,
                'min_confidence': 0.7
            }
        }
        
        self.pose_confidence_threshold = 0.5
        self.angle_tolerance = 15
    
    def classify_pose(self, landmarks) -> Dict[str, Any]:
        """Classify the current pose based on landmarks"""
        if not landmarks:
            return {'pose_name': 'Unknown', 'confidence': 0.0, 'accuracy': 0.0}
        
        pose_metrics = self._extract_pose_metrics(landmarks)
        pose_scores = {}
        
        for pose_name, pose_def in self.pose_definitions.items():
            score = self._calculate_pose_score(pose_metrics, pose_def)
            pose_scores[pose_name] = score
        
        best_pose = max(pose_scores.items(), key=lambda x: x[1])
        
        if best_pose[1] >= self.pose_confidence_threshold:
            return {
                'pose_name': best_pose[0],
                'confidence': best_pose[1],
                'accuracy': best_pose[1],
                'metrics': pose_metrics
            }
        else:
            return {
                'pose_name': 'Unknown',
                'confidence': best_pose[1],
                'accuracy': best_pose[1],
                'metrics': pose_metrics
            }
    
    def _extract_pose_metrics(self, landmarks) -> Dict[str, float]:
        """Extract key metrics from pose landmarks"""
        metrics = {}
        
        left_shoulder = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE)
        
        if all([left_shoulder, left_hip, left_knee]):
            metrics['left_hip_angle'] = self._calculate_angle(left_shoulder, left_hip, left_knee)
        
        if all([right_shoulder, right_hip, right_knee]):
            metrics['right_hip_angle'] = self._calculate_angle(right_shoulder, right_hip, right_knee)
        
        if all([left_hip, left_knee]):
            metrics['left_knee_angle'] = self._calculate_angle(left_hip, left_knee, (left_hip[0], left_hip[1] + 0.1))
        
        if all([right_hip, right_knee]):
            metrics['right_knee_angle'] = self._calculate_angle(right_hip, right_knee, (right_hip[0], right_hip[1] + 0.1))
        
        return metrics
    
    def _calculate_pose_score(self, metrics: Dict[str, float], pose_def: Dict[str, Any]) -> float:
        """Calculate confidence score for a specific pose"""
        score = 0.0
        total_checks = 0
        
        if 'hip_angle_range' in pose_def:
            total_checks += 1
            if 'left_hip_angle' in metrics:
                if self._angle_in_range(metrics['left_hip_angle'], pose_def['hip_angle_range']):
                    score += 1.0
        
        if 'knee_angle_range' in pose_def:
            total_checks += 1
            if 'left_knee_angle' in metrics:
                if self._angle_in_range(metrics['left_knee_angle'], pose_def['knee_angle_range']):
                    score += 1.0
        
        if total_checks > 0:
            return score / total_checks
        return 0.0
    
    def _get_landmark_coords(self, landmarks, landmark_id) -> Tuple[float, float, float]:
        """Get coordinates of a specific landmark"""
        landmark = landmarks.landmark[landmark_id]
        return (landmark.x, landmark.y, landmark.z)
    
    def _calculate_angle(self, a: Tuple[float, float, float], 
                        b: Tuple[float, float, float], 
                        c: Tuple[float, float, float]) -> float:
        """Calculate angle between three points"""
        if not all([a, b, c]):
            return 0.0
        
        a = np.array([a[0], a[1]])
        b = np.array([b[0], b[1]])
        c = np.array([c[0], c[1]])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _angle_in_range(self, angle: float, range_tuple: Tuple[float, float]) -> bool:
        """Check if angle is within specified range"""
        min_angle, max_angle = range_tuple
        return min_angle - self.angle_tolerance <= angle <= max_angle + self.angle_tolerance
    
    def draw_pose_info(self, frame, pose_info: Dict[str, Any]):
        """Draw pose information overlay on the frame"""
        pose_name = pose_info.get('pose_name', 'Unknown')
        confidence = pose_info.get('confidence', 0.0)
        accuracy = pose_info.get('accuracy', 0.0)
        
        cv2.putText(frame, f"Pose: {pose_name}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Accuracy: {accuracy:.2f}", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
