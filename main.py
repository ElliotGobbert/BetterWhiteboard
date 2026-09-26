import cv2
from camera_tracker import HandTracker
from gesture_recognizer import is_hand_closed
from canvas_renderer import CanvasRenderer

def main():
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize Modules
    tracker = HandTracker()
    renderer = CanvasRenderer(1280, 720)

    # State variables
    prev_pos = None
    draw_color = (0, 255, 0)  # Green brush (BGR)
    brush_thickness = 8

    print("[INFO] Modular Whiteboard active. Close hand to DRAW. Open hand to PAUSE.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # 1. Process Vision
        results = tracker.process_frame(frame)

        drawing_active = False
        current_pos = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index fingertip coordinates
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                current_pos = (x, y)

                # 2. Check Gestures
                if is_hand_closed(hand_landmarks):
                    drawing_active = True
                    if prev_pos is None:
                        prev_pos = current_pos

                    # 3. Update State & Draw
                    renderer.draw_line(prev_pos, current_pos, draw_color, brush_thickness)
                    prev_pos = current_pos
                else:
                    prev_pos = None  # Reset tracking so lines don't connect across gaps
        else:
            prev_pos = None

        # 4. Render Final Output
        frame = renderer.blend_and_overlay(frame, drawing_active, current_pos)
        cv2.imshow("Jetson AI Whiteboard", frame)

        # Handle Keyboard Input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in [ord("c"), ord("C")]:
            renderer.clear()
            print("[INFO] Canvas cleared!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
