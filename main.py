import cv2
from camera_tracker import HandTracker
from gesture_recognizer import is_hand_closed, count_extended_fingers
from canvas_renderer import CanvasRenderer

def main():
    cap = cv2.VideoCapture(0) # Change to 1 if hitting Continuity Camera bug on Mac
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(max_hands=2)
    renderer = CanvasRenderer(1280, 720)

    prev_pos = None
    left_prev_x = None  # Tracks the left wrist position for swipe velocity
    
    print("[INFO] Modular Whiteboard active.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        results = tracker.process_frame(frame)

        drawing_active = False
        current_pos = None

        if results.multi_hand_landmarks and results.multi_handedness:
            # zip pairs each hand's physical coordinates with its Left/Right classification
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Note: Because cv2.flip mirrors the image, MediaPipe's "Left" label usually aligns with your physical right hand.
                # If the hands feel backwards when you test it, just swap "Right" and "Left" in these statements.
                hand_label = handedness.classification[0].label 

                # 1. RIGHT HAND: Drawing and Erasing
                if hand_label == "Right":
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
                        if prev_pos is None: 
                            prev_pos = current_pos
                        renderer.draw_line(prev_pos, current_pos, color, thick)
                        prev_pos = current_pos
                        
                # 2. LEFT HAND: Swiping to Clear
                elif hand_label == "Left":
                    fingers_up = count_extended_fingers(hand_landmarks)
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    
                    # If hand is open (4+ fingers)
                    if fingers_up >= 4:
                        if left_prev_x is not None:
                            # Calculate horizontal velocity (pixels moved between frames)
                            velocity = wrist_x - left_prev_x
                            
                            # If wrist moved more than 80 pixels in a single frame, trigger clear
                            if abs(velocity) > 80:
                                renderer.clear()
                                print("[INFO] Swipe detected! Canvas cleared.")
                                left_prev_x = None  # Reset to avoid triggering 50 times during one swipe
                                continue
                                
                        left_prev_x = wrist_x
                    else:
                        left_prev_x = None
        else:
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