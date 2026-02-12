"""
Voice Feedback Module
Provides voice guidance and motivation during yoga sessions
"""

import pyttsx3
import threading
import time
from typing import Optional, List
import queue
import random

class VoiceFeedback:
    """Text-to-speech feedback system for yoga instruction"""
    
    def __init__(self):
        """Initialize text-to-speech engine"""
        self.engine = None
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = None
        
        try:
            self.engine = pyttsx3.init()
            self._configure_engine()
            self._start_speech_thread()
            print("Voice feedback system initialized successfully")
        except Exception as e:
            print(f"Voice feedback initialization failed: {e}")
            self.engine = None
    
    def _configure_engine(self):
        """Configure the TTS engine settings"""
        if not self.engine:
            return
        
        try:
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
            self.engine.setProperty('rate', 150)  # Slower rate for yoga (Words per minute)
            self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        except Exception as e:
            print(f"Error configuring voice engine: {e}")
    
    def _start_speech_thread(self):
        """Start background thread for speech processing"""
        if self.speech_thread is None:
            self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.speech_thread.start()
    
    def _speech_worker(self):
        """Background worker for processing speech queue"""
        while True:
            try:
                # Get text from queue (blocks for 1 sec max to allow checking shutdown)
                text = self.speech_queue.get(timeout=1)
                
                if text is None:  # Shutdown signal
                    break
                
                self.is_speaking = True
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.is_speaking = False
                
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech error: {e}")
                self.is_speaking = False
    
    def speak(self, text: str, priority: bool = False):
        """
        Speak the given text (non-blocking).
        
        Args:
            text: The string to speak.
            priority: If True, clears the current queue to speak this immediately.
        """
        if not self.engine:
            # Fallback log if voice is broken
            print(f"[Voice Simulated]: {text}")
            return
        
        if priority:
            # Clear queue and speak immediately for important messages
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
        
        self.speech_queue.put(text)
    
    def speak_pose_instruction(self, pose_name: str, duration: int = 30):
        """Provide specific instruction for a yoga pose"""
        instructions = {
            'Tree Pose': f"Let's practice Tree Pose. Stand tall on one leg, place your other foot on your inner thigh or calf. Avoid the knee joint. Hold for {duration} seconds.",
            'Warrior II': f"Moving to Warrior Two. Step your feet wide, bend your front knee to 90 degrees, and extend your arms out to the sides. Gaze over your front hand.",
            'Downward Dog': f"Transition to Downward Dog. Plant your hands shoulder-width apart, lift your hips high, and press your heels toward the mat. Lengthen your spine."
        }
        
        instruction = instructions.get(pose_name, f"Hold {pose_name} for {duration} seconds. Focus on your form and breathing.")
        self.speak(instruction)
    
    def speak_pose_feedback(self, pose_name: str, accuracy: float, duration: float):
        """Provide feedback after completing a pose"""
        if accuracy >= 80.0:  # Assuming accuracy is 0-100 or 0.8-1.0
            feedback = f"Excellent {pose_name}! Your form was perfect."
        elif accuracy >= 60.0:
            feedback = f"Good {pose_name}. Keep working on your alignment."
        elif accuracy >= 40.0:
            feedback = f"Nice effort on {pose_name}. Try to hold it steadier next time."
        else:
            feedback = f"Keep practicing {pose_name}. Focus on the basics and don't give up."
        
        self.speak(feedback)
    
    def speak_motivation(self):
        """Provide motivational messages during practice"""
        motivational_messages = [
            "You're doing great! Keep breathing and stay present.",
            "Listen to your body and honor its limits.",
            "You're building strength and flexibility.",
            "Remember, yoga is a journey, not a destination.",
            "Take a deep breath and feel the energy flowing through you.",
            "Focus on your breath.",
            "Relax your shoulders."
        ]
        
        message = random.choice(motivational_messages)
        self.speak(message)
    
    def speak_session_start(self):
        """Welcome message for starting a yoga session"""
        welcome_message = "Welcome to your AI yoga session! I'm here to guide you. Let's begin by finding a comfortable standing position."
        self.speak(welcome_message, priority=True)
    
    def speak_session_end(self):
        """Summary message for ending a yoga session"""
        summary = "Session ended. Wonderful work today. Namaste."
        self.speak(summary, priority=True)

    def stop_speaking(self):
        """Stop any current speech"""
        if self.engine and self.is_speaking:
            self.engine.stop()
            self.is_speaking = False
            # Clear queue
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
    
    def cleanup(self):
        """Clean up resources"""
        if self.engine:
            self.stop_speaking()
            self.speech_queue.put(None)  # Shutdown signal
            if self.speech_thread:
                self.speech_thread.join(timeout=2)
            self.engine = None