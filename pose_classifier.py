"""
Pose Classifier Module (Strict Mode)
Logic for angle calculation, pose scoring, and feedback generation.
Now includes stricter tolerances (10 degrees) and multi-point checks.
"""

import math
import numpy as np

class PoseClassifier:
    def __init__(self):
        # "Strict Mode" Definitions: Tolerance is tight (+/- 10 degrees)
        # We added 'armpit_angle' and 'foot_spread' to stop pose confusion.
        self.pose_definitions = {
            'Tree Pose': {
                'description': "Stand on one leg, foot on thigh. Hands in prayer.",
                # One leg straight (170-180), One leg bent (< 90 usually)
                # But since side varies, we check for ONE straight knee
                'conditions': {
                    'min_straight_knee': 170, # At least one leg must be straight
                    'min_foot_lift_y': 0.05,  # One foot must be higher than the other
                },
                'correction_checks': {
                    'hip_alignment': "Keep your hips level.",
                    'standing_leg': "Straighten your standing leg."
                }
            },
            'Warrior II': {
                'description': "Lunge deep, arms wide, gaze forward.",
                'knee_angle_range': (80, 100),       # Front knee bent 90 (+/- 10)
                'back_knee_angle_range': (170, 180), # Back leg straight
                'elbow_angle_range': (170, 180),     # Arms straight
                'armpit_angle_range': (80, 100),     # Arms horizontal (90 deg from body)
                'conditions': {
                    'min_foot_spread_x': 0.15        # Feet must be wide apart
                }
            },
            'Triangle Pose': {
                'description': "Legs straight, reach down, arm up.",
                'knee_angle_range': (170, 180),      # BOTH legs straight (Crucial diff from Warrior II)
                'armpit_angle_range': (80, 100),     # Arms expanded
                'conditions': {
                    'min_foot_spread_x': 0.15
                }
            },
            'Chair Pose': {
                'description': "Sit back, arms up.",
                'knee_angle_range': (80, 120),       # Knees bent
                'hip_angle_range': (80, 120),        # Hips bent
                'armpit_angle_range': (150, 180),    # Arms raised UP (Crucial diff from simple squat)
                'conditions': {
                    'max_foot_spread_x': 0.15        # Feet should be relatively close
                }
            },
            'Downward Dog': {
                'description': "Inverted V shape.",
                'hip_angle_range': (70, 100),        # Sharp hip flexion
                'knee_angle_range': (170, 180),      # Legs straight
                'elbow_angle_range': (170, 180),     # Arms straight
                'shoulder_angle_range': (160, 180),  # Arms inline with torso (Open shoulders)
                'conditions': {
                    'hands_on_ground': True
                }
            },
            'Goddess Pose': {
                'description': "Wide squat, elbows bent.",
                'knee_angle_range': (80, 110),       # Deep squat
                'elbow_angle_range': (80, 100),      # Cactus arms (bent 90)
                'conditions': {
                    'min_foot_spread_x': 0.15        # Wide stance
                }
            },
            'Warrior I': {
                'description': "High lunge, arms up.",
                'knee_angle_range': (80, 100),       # Front knee bent
                'back_knee_angle_range': (170, 180), # Back leg straight
                'armpit_angle_range': (150, 180),    # Arms UP (Diff from Warrior II)
            }
        }

    
    def _extract_pose_metrics(self, lm):
        """Calculate detailed geometric features from landmarks."""
        # Helper: Get coordinates
        def get_coords(idx):
            return [lm[idx].x, lm[idx].y, lm[idx].z] if idx < len(lm) else None

        # Landmarks (MediaPipe indices)
        # 11,12: Shoulders | 13,14: Elbows | 15,16: Wrists
        # 23,24: Hips      | 25,26: Knees  | 27,28: Ankles
        
        l_sh, r_sh = get_coords(11), get_coords(12)
        l_el, r_el = get_coords(13), get_coords(14)
        l_wr, r_wr = get_coords(15), get_coords(16)
        l_hip, r_hip = get_coords(23), get_coords(24)
        l_kn, r_kn = get_coords(25), get_coords(26)
        l_ank, r_ank = get_coords(27), get_coords(28)

        if not l_ank or not r_ank: return {} # Safety check

        metrics = {}

        # 1. Joint Angles (Average of Left/Right for symmetry, or specific)
        metrics['knee_angle'] = (self._calculate_angle(l_hip, l_kn, l_ank) + self._calculate_angle(r_hip, r_kn, r_ank)) / 2
        metrics['hip_angle'] = (self._calculate_angle(l_sh, l_hip, l_kn) + self._calculate_angle(r_sh, r_hip, r_kn)) / 2
        metrics['elbow_angle'] = (self._calculate_angle(l_sh, l_el, l_wr) + self._calculate_angle(r_sh, r_el, r_wr)) / 2
        
        # 2. Armpit Angle (Body to Upper Arm) - Crucial for "Arms Up" vs "Arms Side"
        # Angle between: Hip -> Shoulder -> Elbow
        metrics['armpit_angle'] = (self._calculate_angle(l_hip, l_sh, l_el) + self._calculate_angle(r_hip, r_sh, r_el)) / 2

        # 3. Shoulder Openness (for Down Dog) - Wrist -> Shoulder -> Hip
        metrics['shoulder_angle'] = (self._calculate_angle(l_wr, l_sh, l_hip) + self._calculate_angle(r_wr, r_sh, r_hip)) / 2

        # 4. Special Logic: Asymmetric Poses (Warrior, Tree)
        # We define "Straightest Leg" and "Bent Leg" dynamically
        left_knee_ang = self._calculate_angle(l_hip, l_kn, l_ank)
        right_knee_ang = self._calculate_angle(r_hip, r_kn, r_ank)
        
        metrics['straight_knee_angle'] = max(left_knee_ang, right_knee_ang)
        metrics['bent_knee_angle'] = min(left_knee_ang, right_knee_ang)

        # 5. Distances / Relative Positions (Normalized)
        # Foot Spread (X-axis distance)
        metrics['foot_spread_x'] = abs(l_ank[0] - r_ank[0])
        
        # Foot Lift (Y-axis difference) - For Tree Pose
        metrics['foot_lift_y'] = abs(l_ank[1] - r_ank[1])
        
        # Hand Height (Are hands above head?) - Lower Y is higher in image coords
        # Average Shoulder Y vs Average Wrist Y
        avg_sh_y = (l_sh[1] + r_sh[1]) / 2
        avg_wr_y = (l_wr[1] + r_wr[1]) / 2
        metrics['hands_above_shoulders'] = avg_wr_y < avg_sh_y # True if hands are up

        return metrics
    
    
    def classify_pose(self, landmarks):
        """
        Input: normalized landmarks (List of {x, y, z})
        Output: {pose_name, accuracy, metrics}
        """
        metrics = self._extract_pose_metrics(landmarks)
        
        best_pose = None
        highest_score = 0.0

        for pose_name, criteria in self.pose_definitions.items():
            score = self._calculate_pose_score(metrics, criteria)
            if score > highest_score:
                highest_score = score
                best_pose = pose_name

        # Threshold to avoid guessing random poses
        if highest_score < 40.0:
            return {"pose_name": "Unknown", "accuracy": 0.0, "metrics": metrics}

        return {
            "pose_name": best_pose,
            "accuracy": round(highest_score, 1),
            "metrics": metrics
        }

    

    def _calculate_pose_score(self, metrics, criteria):
        """
        Score a pose based on how many criteria are met.
        Strict 10-degree tolerance is enforced in the dictionary definition ranges.
        """
        score = 0
        total_weight = 0

        # 1. Check Angle Ranges
        # If 'knee_angle_range' exists, check against metrics['knee_angle']
        # Special case: Warriors/Tree use specific 'bent_knee_angle' etc.
        
        checks = [
            ('knee_angle_range', 'knee_angle'),
            ('hip_angle_range', 'hip_angle'),
            ('elbow_angle_range', 'elbow_angle'),
            ('armpit_angle_range', 'armpit_angle'),
            ('shoulder_angle_range', 'shoulder_angle'),
            ('back_knee_angle_range', 'straight_knee_angle')
        ]
        
        # Override for Asymmetric poses (Warrior/Tree)
        if 'back_knee_angle_range' in criteria: 
             # If pose has a 'back leg', map 'knee_angle_range' to the BENT leg
             # and 'back_knee_angle_range' to the STRAIGHT leg
             score_map = {
                 'knee_angle_range': 'bent_knee_angle',
                 'back_knee_angle_range': 'straight_knee_angle'
             }
        else:
             score_map = {}

        for def_key, metric_key in checks:
            if def_key in criteria:
                target_metric = score_map.get(def_key, metric_key)
                min_v, max_v = criteria[def_key]
                actual = metrics.get(target_metric, 0)
                
                total_weight += 20 # Angles are worth 20 points
                
                # Strict Scoring
                if min_v <= actual <= max_v:
                    score += 20
                elif min_v - 15 <= actual <= max_v + 15:
                    score += 10 # Partial credit for being close
                
        # 2. Check Conditions (Booleans/Thresholds)
        if 'conditions' in criteria:
            conds = criteria['conditions']
            
            # A. Foot Spread (Wide Stance)
            if 'min_foot_spread_x' in conds:
                total_weight += 15
                if metrics['foot_spread_x'] > conds['min_foot_spread_x']:
                    score += 15
            
            # B. Feet Together
            if 'max_foot_spread_x' in conds:
                total_weight += 15
                if metrics['foot_spread_x'] < conds['max_foot_spread_x']:
                    score += 15

            # C. Foot Lift (Tree Pose)
            if 'min_foot_lift_y' in conds:
                total_weight += 25 # High value for distinguishing feature
                if metrics['foot_lift_y'] > conds['min_foot_lift_y']:
                    score += 25

            # D. Straight Knee Check (Tree)
            if 'min_straight_knee' in conds:
                total_weight += 15
                if metrics['straight_knee_angle'] > conds['min_straight_knee']:
                    score += 15

            # E. Hands on Ground (Down Dog) - Wrist Y approx same as Ankle Y
            if 'hands_on_ground' in conds:
                # This is hard to calculate exactly without ground plane, 
                # but we can check if wrists are below knees (in Y coord)
                total_weight += 15
                # Note: Y increases downwards. Wrist Y > Knee Y means Wrists are lower
                # We'll use a rough heuristic
                pass 

        if total_weight == 0: return 0
        return (score / total_weight) * 100.0

    def _calculate_angle(self, a, b, c):
        """Calculate angle at point b (in degrees) given 3D coordinates [x, y, z]"""
        if not a or not b or not c: return 0
        
        # Convert to numpy arrays
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Vectors BA and BC
        ba = a - b
        bc = c - b
        
        # Cosine rule: u . v = |u||v| cos(theta)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def get_correction_feedback(self, pose_name, metrics):
        """Generate specific text feedback based on strict failures."""
        if pose_name not in self.pose_definitions: return None
        
        def_range = self.pose_definitions[pose_name]
        
        # 1. Knee Checks
        if 'knee_angle_range' in def_range:
            # Handle asymmetry
            target_metric = 'bent_knee_angle' if 'back_knee_angle_range' in def_range else 'knee_angle'
            actual = metrics.get(target_metric, 180)
            target_min, target_max = def_range['knee_angle_range']
            
            if actual > target_max + 10: return "Bend your knees deeper."
            if actual < target_min - 10: return "Don't squat so low."
            
        # 2. Arm/Shoulder Checks
        if 'armpit_angle_range' in def_range:
            actual = metrics.get('armpit_angle', 0)
            t_min, t_max = def_range['armpit_angle_range']
            
            if t_min > 140 and actual < 130: return "Reach your arms up higher."
            if t_max < 110 and actual > 120: return "Lower your arms to shoulder height."

        # 3. Tree Pose Foot Check
        if pose_name == 'Tree Pose' and metrics['foot_lift_y'] < 0.05:
            return "Lift your foot higher onto your thigh."

        return None

    def draw_pose_info(self, image, pose_info):
        """Optional: Helper to draw text on frame (kept simple for backend logic)"""
        return image