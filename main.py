import cv2
from camera_tracker import HandTracker
from gesture_recognizer import is_hand_closed, count_extended_fingers
from canvas_renderer import CanvasRenderer

def main():
    # Select the default camera. A different index may be needed if another
    # camera (such as Continuity Camera) is selected by the operating system.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(max_hands=2)
    renderer = CanvasRenderer(1280, 720)

    # These persist across frames so drawing is continuous and swipes can be
    # measured as a frame-to-frame wrist displacement.
    prev_pos = None
    left_prev_x = None
    
    print("[INFO] Modular Whiteboard active.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Mirror the preview so it behaves like looking into a mirror.
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        results = tracker.process_frame(frame)

        drawing_active = False
        current_pos = None

        if results.multi_hand_landmarks and results.multi_handedness:
            # Pair each set of landmarks with MediaPipe's left/right label.
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Mirroring reverses the apparent side of a hand in the preview.
                # Swap these labels if the controls feel reversed on this camera.
                hand_label = handedness.classification[0].label 

                # The index fingertip is the cursor for drawing and erasing.
                if hand_label == "Right":
                    # MediaPipe landmarks are normalized (0-1); OpenCV drawing
                    # functions require pixel coordinates.
                    x = int(hand_landmarks.landmark[8].x * w)
                    y = int(hand_landmarks.landmark[8].y * h)
                    current_pos = (x, y)
                    
                    fingers_up = count_extended_fingers(hand_landmarks)
                    
                    if is_hand_closed(hand_landmarks):
                        drawing_active = "DRAW"
                        color, thick = (0, 255, 0), 8
                    elif fingers_up == 2:
                        drawing_active = "ERASE"
                        color, thick = (0, 0, 0), 60  # Black color effectively erases
                    else:
                        drawing_active = False
                        prev_pos = None

                    if drawing_active:
                        # Start a new stroke at the first active position, then
                        # connect later positions to create a continuous line.
                        if prev_pos is None: 
                            prev_pos = current_pos
                        renderer.draw_line(prev_pos, current_pos, color, thick)
                        prev_pos = current_pos
                        
                # An open opposite hand acts as a horizontal clear gesture.
                elif hand_label == "Left":
                    fingers_up = count_extended_fingers(hand_landmarks)
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    
                    # Require an open hand to avoid clearing while repositioning.
                    if fingers_up >= 4:
                        if left_prev_x is not None:
                            # Approximate horizontal velocity from consecutive frames.
                            velocity = wrist_x - left_prev_x
                            
                            # A large one-frame movement is treated as a swipe.
                            if abs(velocity) > 80:
                                renderer.clear()
                                print("[INFO] Swipe detected! Canvas cleared.")
                                # Reset so one swipe cannot clear repeatedly.
                                left_prev_x = None
                                continue
                                
                        left_prev_x = wrist_x
                    else:
                        left_prev_x = None
        else:
            # Do not connect a future detected hand to a stale cursor position.
            prev_pos = None
            left_prev_x = None

        frame = renderer.blend_and_overlay(frame, drawing_active, current_pos)
        cv2.imshow("Jetson AI Whiteboard", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
