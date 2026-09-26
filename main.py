import cv2
import time

from camera_tracker import HandTracker
from gesture_recognizer import get_gesture
from canvas_renderer import CanvasRenderer
from game_manager import GameManager, GameState


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CAMERA_INDEX = 0

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

ROUND_DURATION = 30


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def draw_word_on_screen(
    frame,
    word,
    remaining_seconds
):
    """
    Draws the secret word and timer on the drawing screen.
    """

    height, width, _ = frame.shape

    # Secret word panel

    cv2.rectangle(
        frame,
        (width // 2 - 250, 10),
        (width // 2 + 250, 80),
        (40, 40, 40),
        -1
    )

    word_text = f"DRAW: {word.upper()}"

    text_size = cv2.getTextSize(
        word_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        2
    )[0]

    text_x = (
        width // 2
        - text_size[0] // 2
    )

    cv2.putText(
        frame,
        word_text,
        (text_x, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Countdown circle

    if remaining_seconds <= 10:
        timer_color = (0, 0, 255)
    else:
        timer_color = (0, 200, 255)

    cv2.circle(
        frame,
        (width - 80, 70),
        45,
        timer_color,
        4
    )

    timer_text = str(remaining_seconds)

    text_size = cv2.getTextSize(
        timer_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        3
    )[0]

    timer_x = (
        width - 80
        - text_size[0] // 2
    )

    timer_y = (
        70
        + text_size[1] // 2
    )

    cv2.putText(
        frame,
        timer_text,
        (timer_x, timer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        timer_color,
        3,
        cv2.LINE_AA
    )

    return frame


def create_reveal_screen(
    drawing,
    word,
    width,
    height
):
    """
    Creates the standalone drawing screen.

    The actual drawing is shown without the camera feed.
    """

    header_height = 100

    screen = cv2.copyMakeBorder(
        drawing,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(245, 245, 245)
    )

    # Title

    cv2.putText(
        screen,
        "TIME'S UP!",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 30),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        screen,
        "Press SPACE for the next round",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 80, 80),
        2,
        cv2.LINE_AA
    )

    # Do NOT show the answer here.
    #
    # This is important because this screen is intended
    # for the other players to guess the drawing.

    return screen


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

def main():

    print("=" * 50)
    print("Pictionary AI Whiteboard")
    print("=" * 50)

    # -----------------------------------------------------
    # Camera
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        print(
            "[ERROR] Could not open camera."
        )

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    # -----------------------------------------------------
    # Components
    # -----------------------------------------------------

    tracker = HandTracker(
        max_hands=2,
        min_detect_conf=0.8,
        min_track_conf=0.5
    )

    renderer = CanvasRenderer(
        FRAME_WIDTH,
        FRAME_HEIGHT
    )

    game = GameManager(
        round_duration=ROUND_DURATION
    )

    # -----------------------------------------------------
    # Drawing state
    # -----------------------------------------------------

    previous_position = None

    left_previous_x = None

    # Prevent repeated swipe detection.

    swipe_cooldown_until = 0

    print(
        "[INFO] Game started."
    )

    print(
        "[INFO] Draw the displayed word using your "
        "right hand."
    )

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            print(
                "[WARNING] Failed to read camera frame."
            )

            continue

        # Mirror camera like a selfie camera.

        frame = cv2.flip(
            frame,
            1
        )

        frame_height, frame_width, _ = frame.shape

        # =================================================
        # DRAWING STATE
        # =================================================

        if game.is_drawing():

            remaining_seconds = (
                game.get_remaining_seconds()
            )

            # ---------------------------------------------
            # Timer expiration
            # ---------------------------------------------

            if game.is_time_up():

                print(
                    "[GAME] 30 seconds elapsed."
                )

                final_drawing = (
                    renderer.get_drawing()
                )

                game.finish_round(
                    final_drawing
                )

                previous_position = None
                left_previous_x = None

                continue

            # ---------------------------------------------
            # Hand tracking
            # ---------------------------------------------

            results = tracker.process_frame(
                frame
            )

            drawing_active = "PAUSE"

            current_position = None

            if (
                results.multi_hand_landmarks
                and results.multi_handedness
            ):

                for (
                    hand_landmarks,
                    handedness
                ) in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    hand_label = (
                        handedness
                        .classification[0]
                        .label
                    )

                    # =====================================
                    # RIGHT HAND
                    # =====================================

                    if hand_label == "Right":

                        index_tip = (
                            hand_landmarks
                            .landmark[8]
                        )

                        x = int(
                            index_tip.x
                            * frame_width
                        )

                        y = int(
                            index_tip.y
                            * frame_height
                        )

                        # Keep pointer inside canvas.

                        x = max(
                            0,
                            min(
                                frame_width - 1,
                                x
                            )
                        )

                        y = max(
                            0,
                            min(
                                frame_height - 1,
                                y
                            )
                        )

                        current_position = (
                            x,
                            y
                        )

                        gesture = get_gesture(
                            hand_landmarks
                        )

                        # ---------------------------------
                        # DRAW
                        # ---------------------------------

                        if gesture == "DRAW":

                            drawing_active = "DRAW"

                            if previous_position is None:

                                previous_position = (
                                    current_position
                                )

                            renderer.draw_line(
                                previous_position,
                                current_position,
                                color=(57, 255, 20),
                                thickness=8
                            )

                            previous_position = (
                                current_position
                            )

                        # ---------------------------------
                        # ERASE
                        # ---------------------------------

                        elif gesture == "ERASE":

                            drawing_active = "ERASE"

                            renderer.erase(
                                current_position,
                                radius=40
                            )

                            # Do not connect drawing
                            # after erasing.

                            previous_position = None

                        # ---------------------------------
                        # PAUSE
                        # ---------------------------------

                        else:

                            drawing_active = "PAUSE"

                            previous_position = None

                    # =====================================
                    # LEFT HAND
                    # =====================================

                    elif hand_label == "Left":

                        fingers_up = (
                            0
                        )

                        # Count extended fingers.

                        tip_indices = [
                            8,
                            12,
                            16,
                            20
                        ]

                        pip_indices = [
                            6,
                            10,
                            14,
                            18
                        ]

                        for (
                            tip_idx,
                            pip_idx
                        ) in zip(
                            tip_indices,
                            pip_indices
                        ):

                            tip = (
                                hand_landmarks
                                .landmark[tip_idx]
                            )

                            pip = (
                                hand_landmarks
                                .landmark[pip_idx]
                            )

                            if tip.y < pip.y:

                                fingers_up += 1

                        wrist = (
                            hand_landmarks
                            .landmark[0]
                        )

                        wrist_x = int(
                            wrist.x
                            * frame_width
                        )

                        # ---------------------------------
                        # Open hand swipe
                        # ---------------------------------

                        if fingers_up >= 4:

                            current_time = time.monotonic()

                            if (
                                left_previous_x
                                is not None
                                and current_time
                                >= swipe_cooldown_until
                            ):

                                velocity = (
                                    wrist_x
                                    - left_previous_x
                                )

                                # Swipe threshold.

                                if abs(velocity) > 60:

                                    renderer.clear()

                                    previous_position = None

                                    left_previous_x = None

                                    swipe_cooldown_until = (
                                        current_time
                                        + 0.8
                                    )

                                    print(
                                        "[GAME] "
                                        "Canvas cleared."
                                    )

                            if (
                                current_time
                                >= swipe_cooldown_until
                            ):

                                left_previous_x = (
                                    wrist_x
                                )

                        else:

                            left_previous_x = None

            else:

                previous_position = None
                left_previous_x = None

            # ---------------------------------------------
            # Render drawing interface
            # ---------------------------------------------

            frame = renderer.blend_and_overlay(
                frame,
                drawing_active,
                current_position,
                remaining_seconds
            )

            frame = draw_word_on_screen(
                frame,
                game.get_word(),
                remaining_seconds
            )

            cv2.imshow(
                "Pictionary",
                frame
            )

        # =================================================
        # REVEAL STATE
        # =================================================

        elif game.is_reveal():

            reveal_screen = create_reveal_screen(
                game.final_drawing,
                game.get_word(),
                FRAME_WIDTH,
                FRAME_HEIGHT
            )

            cv2.imshow(
                "Pictionary",
                reveal_screen
            )

        # -------------------------------------------------
        # Keyboard controls
        # -------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # ESC
        if key == 27:

            print(
                "[INFO] Exiting game."
            )

            break

        # SPACE = next round

        if (
            key == ord(" ")
            and game.is_reveal()
        ):

            renderer.clear()

            game.start_new_round()

            previous_position = None
            left_previous_x = None

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    tracker.close()

    cap.release()

    cv2.destroyAllWindows()

    print(
        "[INFO] Pictionary closed."
    )


if __name__ == "__main__":
    main()