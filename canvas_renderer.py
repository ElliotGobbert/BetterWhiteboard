import cv2
import numpy as np

class CanvasRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def draw_line(self, start_pos, end_pos, color, thickness):
        """Draws a line on the persistent canvas."""
        cv2.line(self.canvas, start_pos, end_pos, color, thickness)

    def clear(self):
        """Resets the canvas to a blank black screen."""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def blend_and_overlay(self, frame, drawing_active, current_pos):
        """Blends the drawing canvas with the live frame and adds UI overlays."""
        overlay_alpha = 0.7

        # Keep the camera visible while making the drawing semi-transparent.
        # Black pixels in the canvas are treated as fully transparent so they don't
        # obscure the live image, while colored strokes are blended in softly.
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        overlay = self.canvas.copy()
        overlay[mask == 0] = [0, 0, 0]

        frame = cv2.addWeighted(frame, 1.0 - overlay_alpha, overlay, overlay_alpha, 0)

        # Visual indicator for fingertip
        if current_pos:
            if drawing_active == "DRAW":
                pointer_color = (0, 255, 0)  # Green
            elif drawing_active == "ERASE":
                pointer_color = (0, 0, 255)  # Red
            else:
                pointer_color = (255, 0, 0)  # Blue
            cv2.circle(frame, current_pos, 12, pointer_color, -1)

        # Status overlay text
        if drawing_active == "DRAW":
            status_text = "STATUS: DRAWING (Fist)"
            status_color = (0, 255, 0)
        elif drawing_active == "ERASE":
            status_text = "STATUS: ERASING (2 Fingers)"
            status_color = (0, 0, 255)
        else:
            status_text = "STATUS: PAUSED (Open Hand)"
            status_color = (255, 0, 0)

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, "Controls: Left Hand Swipe = Clear | [ESC] Exit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame