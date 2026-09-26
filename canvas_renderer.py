import cv2
import numpy as np


class CanvasRenderer:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = self._create_blank_canvas()

    def _create_blank_canvas(self):
        """
        Creates a white drawing canvas.
        OpenCV uses BGR, so 255,255,255 is white.
        """
        return np.ones(
            (self.height, self.width, 3),
            dtype=np.uint8
        ) * 255

    def draw_line(
        self,
        start_pos,
        end_pos,
        color=(57, 255, 20),  # Neon green
        thickness=8
    ):
        """Draws a line onto the persistent canvas."""
        cv2.line(
            self.canvas,
            start_pos,
            end_pos,
            color,
            thickness,
            lineType=cv2.LINE_AA
        )

    def erase(
        self,
        position,
        radius=50
    ):
        """Erases a circular region of the drawing."""
        cv2.circle(
            self.canvas,
            position,
            radius,
            (255, 255, 255),
            -1
        )

    def clear(self):
        """Resets the canvas."""
        self.canvas = self._create_blank_canvas()

    def get_drawing(self):
        """
        Returns a copy of the canvas.
        This is the image used during the final reveal.
        """
        return self.canvas.copy()

    def has_drawing(self):
        """
        Determines whether anything has been drawn.
        """
        difference = cv2.absdiff(
            self.canvas,
            np.ones_like(self.canvas) * 255
        )

        return np.any(difference > 10)

    def blend_and_overlay(
        self,
        frame,
        drawing_active,
        current_pos,
        remaining_seconds
    ):
        """
        Displays the drawing over the live camera feed.

        The camera is darkened so the neon-green drawing
        is much more visually prominent.
        """

        # ------------------------------------------------
        # Camera / Drawing Visibility
        # ------------------------------------------------

        # How visible the camera should be.
        # Lower = darker camera.
        camera_alpha = 0.25

        # How opaque the drawing should be.
        drawing_alpha = 0.95

        # Darken the entire camera feed.
        frame = (
            frame.astype(np.float32) * camera_alpha
        ).astype(np.uint8)

        # ------------------------------------------------
        # Drawing Mask
        # ------------------------------------------------

        gray = cv2.cvtColor(
            self.canvas,
            cv2.COLOR_BGR2GRAY
        )

        # White canvas pixels are ignored.
        drawing_mask = cv2.inRange(
            gray,
            0,
            245
        )

        # ------------------------------------------------
        # Put Drawing Over Camera
        # ------------------------------------------------

        canvas_pixels = self.canvas[drawing_mask > 0]

        if len(canvas_pixels) > 0:

            frame[drawing_mask > 0] = (
                frame[drawing_mask > 0] * (1 - drawing_alpha)
                + canvas_pixels * drawing_alpha
            ).astype(np.uint8)

        # ------------------------------------------------
        # Fingertip Indicator
        # ------------------------------------------------

        if current_pos is not None:

            if drawing_active == "DRAW":
                pointer_color = (57, 255, 20)  # Neon green

            elif drawing_active == "ERASE":
                pointer_color = (0, 0, 255)

            else:
                pointer_color = (255, 120, 0)

            cv2.circle(
                frame,
                current_pos,
                12,
                pointer_color,
                -1
            )

        # ------------------------------------------------
        # Timer
        # ------------------------------------------------

        if remaining_seconds <= 10:
            timer_color = (0, 0, 255)
        else:
            timer_color = (255, 255, 255)

        timer_text = f"TIME: {remaining_seconds}s"

        cv2.putText(
            frame,
            timer_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            timer_color,
            3,
            cv2.LINE_AA
        )

        # ------------------------------------------------
        # Gesture Status
        # ------------------------------------------------

        if drawing_active == "DRAW":
            status_text = "DRAWING - Fist"
            status_color = (57, 255, 20)

        elif drawing_active == "ERASE":
            status_text = "ERASING - 2 Fingers"
            status_color = (0, 0, 255)

        else:
            status_text = "PAUSED - Open Hand"
            status_color = (255, 120, 0)

        cv2.putText(
            frame,
            status_text,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA
        )

        # ------------------------------------------------
        # Controls
        # ------------------------------------------------

        controls = "Left Hand Swipe = Clear | ESC = Exit"

        cv2.putText(
            frame,
            controls,
            (20, self.height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame

    def render_final_drawing(
        self,
        drawing,
        word
    ):
        """
        Creates the standalone image shown after the timer ends.

        The camera is NOT included.
        """

        image = drawing.copy()

        # Add a small header area.
        header_height = 70

        output = np.ones(
            (
                self.height + header_height,
                self.width,
                3
            ),
            dtype=np.uint8
        ) * 255

        output[
            header_height:,
            :
        ] = image

        cv2.putText(
            output,
            "DRAWING COMPLETE",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (30, 30, 30),
            2,
            cv2.LINE_AA
        )

        return output