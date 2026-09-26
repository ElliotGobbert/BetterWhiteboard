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
        # Blend drawing canvas with live frame
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, self.canvas)

        # Visual indicator for fingertip
        if current_pos:
            pointer_color = (0, 0, 255) if drawing_active else (255, 0, 0)
            cv2.circle(frame, current_pos, 12, pointer_color, -1)

        # Status overlay text
        status_text = "STATUS: DRAWING (Fist)" if drawing_active else "STATUS: PAUSED (Open Hand)"
        status_color = (0, 255, 0) if drawing_active else (255, 0, 0)

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, "Controls: [C] Clear | [ESC] Exit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame
