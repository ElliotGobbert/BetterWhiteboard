# 🎨 BetterWhiteboard

BetterWhiteboard is an AI-powered, touchless virtual whiteboard built for our hackathon. It uses computer vision to track hand gestures, allowing users to draw, erase, and switch colors in mid-air using just a webcam.

The primary hardware target is the **NVIDIA Jetson Orin Nano**, but the software is modularized to run on standard Mac, Windows, and Linux machines for parallel development.

## 🚀 Hackathon End Goals
- **Gesture Controls:** Fist to draw, open hand to pause, peace sign to change colors, swipe to clear the board.
- **Auto-Correction:** Detect drawn shapes (circles, squares) and snap them to perfect geometry.
- **Air-OCR:** Detect handwritten letters and convert them to live digital text.

## 🧩 Modular Architecture
To allow our team to work in parallel without merge conflicts, the codebase is split into four engines:
1. `camera_tracker.py` - Manages webcam feed and MediaPipe hand-tracking.
2. `gesture_recognizer.py` - The math engine. Calculates joint angles and distances to classify gestures.
3. `canvas_renderer.py` - The graphics engine. Handles OpenCV arrays, blending, and UI text overlays.
4. `main.py` - The State Machine. Orchestrates the flow of data between the tracker, recognizer, and renderer.

## 🤝 Contributing
Check out `SETUP.md` for instructions on how to install dependencies and run the project locally.
