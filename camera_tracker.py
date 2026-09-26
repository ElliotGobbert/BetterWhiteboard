import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=1, min_detect_conf=0.8, min_track_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detect_conf,
            min_tracking_confidence=min_track_conf
        )

    def process_frame(self, frame):
        """Converts frame to RGB and processes it through MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results
