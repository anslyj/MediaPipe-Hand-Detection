🎮 Rock Paper Scissors Computer Vision Game
A real-time hand gesture recognition game using OpenCV and MediaPipe to detect rock ✊, paper ✋, and scissors ✌️ gestures through your webcam.
🛠️ Tech Stack

Python, OpenCV, MediaPipe, NumPy

📋 Installation
bashgit clone https://github.com/anslyj/MediaPipe-Hand-Detection.git
cd MediaPipe-Hand-Detection
pip install opencv-python mediapipe numpy
python rock_paper_scissors.py
🎮 How to Play

Position yourself in front of your webcam
Press SPACE to start a round
Show rock ✊, paper ✋, or scissors ✌️ during countdown
Hold gesture steady for detection

Controls: SPACE - Start/Continue | R - Reset scores | Q - Quit
🧠 Features

Real-time hand landmark detection
Custom gesture classification algorithm
Gesture stabilization (700ms hold time)
Live video feedback with hand landmarks
Score tracking and multiple game states

🚨 Troubleshooting

Camera issues: Check webcam connection, try changing cv2.VideoCapture(0) to (1)
Poor detection: Ensure good lighting and clear hand positionin
