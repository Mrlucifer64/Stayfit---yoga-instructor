"""
Pose Classification Module
Classifies yoga poses based on MediaPipe pose landmarks
"""

import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple
import cv2 

class PoseClassifier:
    """Classifies yoga poses based on pose landmarks and geometric calculations"""
    
    def __init__(self):
        """Initialize the pose classifier with yoga pose definitions"""
        self.mp_pose = mp.solutions.pose
        
        # Define target angles and positions for each yoga pose
        self.pose_definitions = {
            'Tree Pose': {
                'hip_angle_range': (150, 180),
                'knee_angle_range': (150, 180), # Standing leg straight
                'min_confidence': 0.7
            },
            'Warrior II': {
                'hip_angle_range': (80, 120),
                'knee_angle_range': (80, 120),
                'min_confidence': 0.7
            },
            'Downward Dog': {
                'hip_angle_range': (60, 100), # Sharper hip angle
                'shoulder_hip_angle': (60, 120),
                'min_confidence': 0.7
            }
        }
        
        self.pose_confidence_threshold = 0.5
        self.angle_tolerance = 20 # Increased slightly for webcam variance
    
    def classify_pose(self, landmarks) -> Dict[str, Any]:
        """Classify the current pose based on landmarks"""
        if not landmarks:
            return {'pose_name': 'Unknown', 'confidence': 0.0, 'accuracy': 0.0}
        
        pose_metrics = self._extract_pose_metrics(landmarks)
        pose_scores = {}
        
        for pose_name, pose_def in self.pose_definitions.items():
            score = self._calculate_pose_score(pose_metrics, pose_def)
            pose_scores[pose_name] = score
        
        if not pose_scores:
             return {'pose_name': 'Unknown', 'confidence': 0.0, 'accuracy': 0.0}

        best_pose = max(pose_scores.items(), key=lambda x: x[1])
        
        # Return result if confidence threshold met
        pose_name = best_pose[0] if best_pose[1] >= self.pose_confidence_threshold else 'Unknown'
        
        return {
            'pose_name': pose_name,
            'confidence': best_pose[1],
            'accuracy': best_pose[1] * 100, # Convert to percentage
            'metrics': pose_metrics
        }
    
    def _extract_pose_metrics(self, landmarks) -> Dict[str, float]:
        """Extract key metrics from pose landmarks"""
        if not landmarks or len(landmarks) < 33:
            return {}
        
        metrics = {}
        
        def get_coord(idx):
            return landmarks[idx.value]

        left_shoulder = get_coord(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_coord(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = get_coord(self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_coord(self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_coord(self.mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_coord(self.mp_pose.PoseLandmark.RIGHT_KNEE)
        
        try:
            metrics['left_hip_angle'] = self._calculate_angle(left_shoulder, left_hip, left_knee)
            metrics['right_hip_angle'] = self._calculate_angle(right_shoulder, right_hip, right_knee)
            # Knee angles (relative to vertical drop)
            metrics['left_knee_angle'] = self._calculate_angle(left_hip, left_knee, (left_hip[0], left_knee[1] + 0.5, left_knee[2]))
        except Exception:
            pass 
            
        return metrics
    
    def _calculate_pose_score(self, metrics: Dict[str, float], pose_def: Dict[str, Any]) -> float:
        """Calculate confidence score for a specific pose"""
        score = 0.0
        total_checks = 0
        
        if 'hip_angle_range' in pose_def:
            total_checks += 1
            angle = metrics.get('left_hip_angle', 0)
            if self._angle_in_range(angle, pose_def['hip_angle_range']):
                score += 1.0
        
        if 'knee_angle_range' in pose_def:
            total_checks += 1
            angle = metrics.get('left_knee_angle', 0)
            if self._angle_in_range(angle, pose_def['knee_angle_range']):
                score += 1.0
        
        return score / total_checks if total_checks > 0 else 0.0

    def _calculate_angle(self, a: Tuple[float, float, float], 
                        b: Tuple[float, float, float], 
                        c: Tuple[float, float, float]) -> float:
        """Calculate angle between three points (2D projection for stability)"""
        a = np.array([a[0], a[1]])
        b = np.array([b[0], b[1]])
        c = np.array([c[0], c[1]])
        
        ba = a - b
        bc = c - b
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _angle_in_range(self, angle: float, range_tuple: Tuple[float, float]) -> bool:
        """Check if angle is within specified range"""
        min_angle, max_angle = range_tuple
        return (min_angle - self.angle_tolerance) <= angle <= (max_angle + self.angle_tolerance)
    

  
    def get_correction_feedback(self, pose_name, metrics):
        """Generate specific feedback to improve pose accuracy"""
        if pose_name not in self.pose_definitions:
            return None
            
        def_range = self.pose_definitions[pose_name]
        feedback = []
        
        # 1. Check Knee Bends (Chair, Goddess)
        if 'knee_angle_range' in def_range:
            angle = metrics.get('knee_angle_range', 0)
            target_min, target_max = def_range['knee_angle_range']
            
            if angle > target_max + 10:
                feedback.append("Bend your knees deeper.")
            elif angle < target_min - 10:
                feedback.append("Lift your hips slightly, don't squat too low.")

        # 2. Check Straight Legs (Triangle, Tree, Warrior II back leg)
        if 'knee_inc_range' in def_range:
            # We can check the knee joint angle if it exists in metrics
            # Even if the pose definition relies on inclination, we want the leg straight
            knee_angle = metrics.get('knee_angle_range', 0) # Average of both or specific
            
            # If the pose expects a straight leg, knee angle should be ~180
            # We allow some flex, but < 160 is definitely bent
            if knee_angle > 0 and knee_angle < 160:
                 return "Straighten your standing leg."

        # 3. Check Hips
        if 'hip_angle_range' in def_range:
            angle = metrics.get('hip_angle_range', 0)
            target_min, target_max = def_range['hip_angle_range']
            
            # For standing poses (Tree, Mountain), hip angle ~180
            if target_min > 150: 
                if angle < target_min - 15:
                    feedback.append("Stand up straighter, open your hips.")
            
            # For bent poses (Chair, Warrior), hip angle ~90
            elif target_max < 130:
                if angle > target_max + 15:
                    feedback.append("Sink your hips lower.")

        # 4. Check Arms (Warrior II, Goddess)
        if 'elbow_angle_range' in def_range: # Goddess
            angle = metrics.get('elbow_angle_range', 0)
            if angle > 130:
                feedback.append("Bend your elbows more for cactus arms.")
        
        elif def_range.get('arms_extended', False): # Warrior II
             # If we had exact elbow angles, we'd check them here
             pass

        if feedback:
            return feedback[0] # Return the most critical correction
        return None