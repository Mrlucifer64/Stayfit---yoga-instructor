"""
Voice Feedback Module 
Manages voice logic and generates text instructions for the client-side to speak.
"""
import queue
import random
import time
from pose_classifier import PoseClassifier



class VoiceFeedback:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.last_motivation_time = 0
        self.last_correction_time = 0
        
        self.instructions = {
            'Tree Pose': "Tree Pose. Stand tall, place one foot on your inner thigh. Avoid the knee.",
            'Warrior II': "Warrior Two. Step wide, bend your front knee to 90 degrees, arms extended.",
            'Downward Dog': "Downward Dog. Hips high, heels down. Press the floor away.",
            'Chair Pose': "Chair Pose. Sit back as if in a chair, reach arms up.",
            'Goddess Pose': "Goddess Pose. Step wide, toes out, sink low. Cactus your arms.",
            'Triangle Pose': "Triangle Pose. Straighten legs, reach forward and tilt down.",
            'Warrior I': "Warrior One. Hips square to front, front knee bent, arms reaching up."
        }
        
        self.transitions = {
            ('Tree Pose', 'Warrior II'): "Step down and widen your stance for Warrior Two.",
            ('Warrior II', 'Triangle Pose'): "Straighten front leg, reach forward for Triangle.",
            ('Triangle Pose', 'Goddess Pose'): "Lift up, turn heels in, sink into Goddess.",
            ('Goddess Pose', 'Chair Pose'): "Step together and sit back into Chair Pose.",
            ('Chair Pose', 'Downward Dog'): "Fold forward and step back to Downward Dog.",
            ('Warrior II', 'Warrior I'): "Turn back foot in, square hips for Warrior One.",
            ('Warrior I', 'Warrior II'): "Open hips to the side for Warrior Two.",
            ('Downward Dog', 'Chair Pose'): "Walk feet to hands and sit back into Chair Pose."
        }
        
        self.motivational_quotes = [
            "Great focus.", "Keep breathing.", "Strong and steady.", 
            "Relax your shoulders.", "Stay present."
        ]

    def speak(self, text: str, priority: bool = False):
        if priority:
            with self.message_queue.mutex:
                self.message_queue.queue.clear()
        self.message_queue.put(text)

    def get_message(self):
        try:
            return self.message_queue.get_nowait()
        except queue.Empty:
            return None

    def speak_pose_instruction(self, pose_name: str, duration: int = 30):
        text = self.instructions.get(pose_name, f"Hold {pose_name}.")
        self.speak(text)

    def speak_pose_transition(self, from_pose: str, to_pose: str):
        text = self.transitions.get((from_pose, to_pose), f"Transition to {to_pose}.")
        self.speak(text)

    def speak_pose_feedback(self, pose_name: str, accuracy: float, duration: float):
        if accuracy >= 90.0: self.speak(f"Perfect {pose_name}!")
        elif accuracy >= 80.0: self.speak(f"Good {pose_name}.")


    def speak_correction(self, feedback_text):
        now = time.time()
        if now - self.last_correction_time > 6.0:
            self.speak(feedback_text)
            self.last_correction_time = now

    def speak_motivation(self):
        now = time.time()
        if now - self.last_motivation_time > 15.0 and self.message_queue.empty():
            self.speak(random.choice(self.motivational_quotes))
            self.last_motivation_time = now

    def speak_session_start(self):
        self.speak("Welcome. Let's begin practicing. ", priority=True)

    def speak_session_end(self):
        pass