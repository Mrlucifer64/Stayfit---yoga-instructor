AI Yoga Instructor 🧘‍♂️🤖
A full-stack AI application that acts as your personal yoga coach. It uses computer vision to analyze your posture in real-time, provides instant voice feedback via the browser, and tracks your long-term progress in a database.

🌟 Features
Real-Time Pose Analysis: Uses Google MediaPipe to detect 33 body landmarks at 30+ FPS on standard CPUs.

Instant Voice Feedback: "Headless" architecture sends text commands to the browser's Web Speech API, ensuring zero-latency voice guidance (e.g., "Straighten your right leg").

Geometric Correction Logic: Custom algorithms calculate joint angles to judge pose accuracy against "Golden Standard" definitions.

Progress Tracking: automatically logs every pose attempted, duration held, and accuracy score to a MySQL database.

Secure Authentication: Complete Login/Register system with password hashing (Bcrypt) and session management.

Responsive UI: Modern dashboard with Dark/Light Mode toggle and live performance graphs.

🏗️ System Architecture
The application follows a Model-View-Controller (MVC) pattern optimized for real-time performance:

Frontend (Client): HTML5/JS handles the video stream display, voice synthesis (TTS), and theme management.

Backend (Server): Flask manages user sessions and routes.

AI Engine: OpenCV captures frames, MediaPipe extracts landmarks, and Python calculates geometry.

Database: MySQL stores user profiles, session logs, and pose metrics.

Code snippet

graph LR
    A[Webcam] --> B(Flask Server)
    B --> C{AI Engine}
    C -->|Landmarks| D[Pose Classifier]
    D -->|Feedback| B
    B -->|JSON| E[Browser UI]
    E -->|Voice| F[User Speakers]
    B -->|Logs| G[(MySQL DB)]
🛠️ Tech Stack
Language: Python 3.10+

Web Framework: Flask

Computer Vision: OpenCV (cv2), MediaPipe

Database: MySQL (Connector: pymysql)

Frontend: HTML, CSS (Inter Font), JavaScript (Fetch API)

Authentication: Flask-Login, Flask-Bcrypt

🚀 Installation & Setup
Prerequisites
Python 3.x installed.

MySQL Server installed and running.

A webcam.

Step 1: Clone the Repository
Bash

git clone https://github.com/yourusername/ai-yoga-instructor.git
cd ai-yoga-instructor



Step 2: Set up the Database
Open your MySQL Command Line or Workbench.

Create the database and tables using the provided schema:

SQL

CREATE DATABASE yoga_app;
USE yoga_app;
-- (Paste the contents of schema.sql here)


Step 3: Configure the App
Open config.py (or create it) and update your database credentials:

Python

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',        # Your MySQL username
    'password': 'password', # Your MySQL password
    'db': 'yoga_app',
}
SECRET_KEY = 'your_secret_key_here'


Step 4: Install Dependencies
Bash

pip install -r requirements.txt


Step 5: Run the Application
Bash

python app.py
The application will start at http://localhost:8080.

📖 Usage Guide
Register/Login: Create an account to track your personal progress.

Start Session: Click the "Start Session" button on the dashboard.

Position Yourself: Stand 6-8 feet away from the camera so your full body is visible.

Practice:

The AI will detect your pose (e.g., Tree Pose).

If your form is incorrect (e.g., bent knee), it will speak a correction.

If your form is good, it will track your "Hold Time."

End Session: Click "End Session" to save your stats and view your summary.

📂 Project Structure
Plaintext

ai_yoga_instructor/
├── app.py                 # Main application entry point & routes
├── pose_detection.py      # MediaPipe wrapper for landmark extraction
├── pose_classifier.py     # Logic for angle calculation & scoring
├── voice_feedback.py      # Voice logic manager (Text generator)
├── progress_tracker.py    # Database interactions (CRUD operations)
├── config.py              # Configuration secrets
├── templates/
│   ├── index.html         # Main Dashboard (Video + Stats)
│   ├── login.html         # Login Page
│   └── register.html      # Registration Page
└── static/
    └── placeholder.jpg    # Default camera image
🔍 Troubleshooting
Camera not opening?

Ensure no other app (Zoom/Teams) is using the camera.

Check cv2.VideoCapture(0) index in app.py. Try changing 0 to 1.

No Voice Feedback?

Check your browser volume.

Ensure you interacted with the page (clicked "Start") as browsers block auto-playing audio.

Database Errors?

Verify your MySQL server is running.

Double-check credentials in config.py.

🔮 Future Scope
Mobile App: Porting the logic to React Native for phone usage.

Gamification: Adding "Streaks" and "Badges" for consistent practice.

Skeleton Overlay: Customizable colors for "Good" vs "Bad" alignment.