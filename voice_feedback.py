"""
Voice Feedback Module
Provides voice guidance and motivation during yoga sessions
"""

import pyttsx3
import threading
import time
from typing import Optional, List
import queue

class VoiceFeedback:
    """Text-to-speech feedback system for yoga instruction"""
    
    def __init__(self):
        """Initialize text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            self._configure_engine()
            self.speech_queue = queue.Queue()
            self.is_speaking = False
            self.speech_thread = None
            self._start_speech_thread()
            print("Voice feedback system initialized successfully")
        except Exception as e:
            print(f"Voice feedback initialization failed: {e}")
            self.engine = None
    
    def _configure_engine(self):
        """Configure the TTS engine settings"""
        if not self.engine:
            return
        
        # Set voice properties
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to find a female voice (often better for yoga instruction)
            female_voice = None
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    female_voice = voice.id
                    break
            
            if female_voice:
                self.engine.setProperty('voice', female_voice)
        
        # Set speech rate and volume
        self.engine.setProperty('rate', 150)  # Words per minute
        self.engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
    
    def _start_speech_thread(self):
        """Start background thread for speech processing"""
        if self.speech_thread is None:
            self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.speech_thread.start()
    
    def _speech_worker(self):
        """Background worker for processing speech queue"""
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                if text is None:  # Shutdown signal
                    break
                
                self.is_speaking = True
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.is_speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech error: {e}")
                self.is_speaking = False
    
    def speak(self, text: str, priority: bool = False):
        """Speak the given text (non-blocking)"""
        if not self.engine:
            print(f"Voice feedback: {text}")
            return
        
        if priority:
            # Clear queue and speak immediately for important messages
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
        
        self.speech_queue.put(text)
    
    def speak_pose_instruction(self, pose_name: str, duration: int = 30):
        """Provide specific instruction for a yoga pose"""
        instructions = {
            'Tree Pose': f"Let's practice Tree Pose. Stand tall, bring one foot to your thigh, and hold for {duration} seconds. Focus on your balance and breathing.",
            'Warrior II': f"Now Warrior Two. Step your feet wide, bend your front knee, and extend your arms. Hold for {duration} seconds while maintaining strong alignment.",
            'Downward Dog': f"Moving into Downward Dog. Press your hands and feet into the ground, lift your hips, and hold for {duration} seconds. Feel the stretch in your hamstrings."
        }
        
        instruction = instructions.get(pose_name, f"Hold {pose_name} for {duration} seconds. Focus on your form and breathing.")
        self.speak(instruction)
    
    def speak_pose_feedback(self, pose_name: str, accuracy: float, duration: float):
        """Provide feedback after completing a pose"""
        if accuracy >= 0.8:
            feedback = f"Excellent {pose_name}! Your form was perfect. You held it for {duration:.1f} seconds with great alignment."
        elif accuracy >= 0.6:
            feedback = f"Good {pose_name}! You held it for {duration:.1f} seconds. Keep working on your alignment for even better form."
        elif accuracy >= 0.4:
            feedback = f"Nice effort on {pose_name}! You held it for {duration:.1f} seconds. With practice, your form will improve significantly."
        else:
            feedback = f"Keep practicing {pose_name}! You held it for {duration:.1f} seconds. Focus on the fundamentals and don't give up."
        
        self.speak(feedback)
    
    def speak_motivation(self):
        """Provide motivational messages during practice"""
        motivational_messages = [
            "You're doing great! Keep breathing and stay present.",
            "Every pose is an opportunity to grow stronger and more flexible.",
            "Listen to your body and honor its limits.",
            "You're building strength, flexibility, and inner peace.",
            "Remember, yoga is a journey, not a destination.",
            "Take a deep breath and feel the energy flowing through you.",
            "You're stronger than you think. Keep going!",
            "Each practice makes you better than the last.",
            "Trust the process and trust yourself.",
            "You're creating positive change in your body and mind."
        ]
        
        import random
        message = random.choice(motivational_messages)
        self.speak(message)
    
    def speak_session_start(self):
        """Welcome message for starting a yoga session"""
        welcome_message = """
        Welcome to your AI yoga session! I'm here to guide you through your practice.
        Let's begin with some deep breathing to center ourselves.
        Take three deep breaths, inhaling through your nose and exhaling through your mouth.
        When you're ready, we'll start with some gentle warm-up poses.
        """
        self.speak(welcome_message)
    
    def speak_session_end(self, total_duration: float, poses_completed: int, average_accuracy: float):
        """Summary message for ending a yoga session"""
        summary = f"""
        Wonderful work! You've completed your yoga session.
        You practiced for {total_duration:.1f} minutes and completed {poses_completed} poses.
        Your average accuracy was {average_accuracy:.1%}.
        Take a moment to thank your body for this practice.
        Remember to stay hydrated and carry this sense of peace with you throughout your day.
        Namaste.
        """
        self.speak(summary)
    
    def speak_pose_transition(self, from_pose: str, to_pose: str):
        """Guide user through pose transitions"""
        transitions = {
            ('Tree Pose', 'Warrior II'): "Now let's transition from Tree Pose to Warrior Two. Step your feet wide and prepare for strength.",
            ('Warrior II', 'Downward Dog'): "From Warrior Two, let's flow into Downward Dog. Step back and press into your hands and feet.",
            ('Downward Dog', 'Tree Pose'): "From Downward Dog, step forward and return to Tree Pose. Find your balance and center."
        }
        
        transition_text = transitions.get((from_pose, to_pose), 
                                       f"Now let's move from {from_pose} to {to_pose}. Take your time with the transition.")
        self.speak(transition_text)
    
    def speak_breathing_guidance(self):
        """Provide breathing guidance during poses"""
        breathing_instructions = [
            "Breathe deeply and steadily. Inhale for four counts, exhale for four counts.",
            "Focus on your breath. Let it guide your movement and hold your pose.",
            "Take slow, controlled breaths. Feel your body relax with each exhalation.",
            "Breathe into any areas of tension. Let your breath help you release and relax.",
            "Maintain steady breathing. Your breath is your anchor in this pose."
        ]
        
        import random
        instruction = random.choice(breathing_instructions)
        self.speak(instruction)
    
    def stop_speaking(self):
        """Stop any current speech"""
        if self.engine and self.is_speaking:
            self.engine.stop()
            self.is_speaking = False
    
    def cleanup(self):
        """Clean up resources"""
        if self.engine:
            self.stop_speaking()
            self.speech_queue.put(None)  # Shutdown signal
            if self.speech_thread:
                self.speech_thread.join(timeout=2)
            self.engine = None
