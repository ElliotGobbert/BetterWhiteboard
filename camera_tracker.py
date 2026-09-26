import cv2
import mediapipe as mp


class HandTracker:
    """Wrap MediaPipe hand detection for use with OpenCV camera frames."""

    def __init__(self, max_hands=2, min_detect_conf=0.8, min_track_conf=0.5):
        # Keep the MediaPipe module so its landmark and result types stay available
        # through the lifetime of this tracker.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detect_conf,
            min_tracking_confidence=min_track_conf
        )

    def process_frame(self, frame):
        """Return MediaPipe's hand landmarks for an OpenCV BGR frame."""
        # OpenCV captures BGR images, while MediaPipe expects RGB input.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results
