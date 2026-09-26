import cv2
import time
import threading
import requests
from camera_tracker import HandTracker
from gesture_recognizer import is_hand_closed, count_extended_fingers
from canvas_renderer import CanvasRenderer

def send_canvas_to_model(canvas):
    """Sends the canvas to the FastAPI server in the background."""
    _, encoded_image = cv2.imencode('.jpg', canvas)
    try:
        response = requests.post(
            "http://localhost:8000/predict", 
            files={"file": ("canvas.jpg", encoded_image.tobytes(), "image/jpeg")}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"[AI Guess] {data['guess']} (Confidence: {data['confidence']}%)")
    except requests.exceptions.RequestException:
        pass  # Server might not be running yet

def main():
    cap = cv2.VideoCapture(0) # Change to 1 if hitting Continuity Camera bug on Mac
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(max_hands=2)
    renderer = CanvasRenderer(1280, 720)

    prev_pos = None
    left_prev_x = None  # Tracks the left wrist position for swipe velocity
    last_predict_time = time.time()
    
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

        # AI Prediction Logic (Triggers every 2.5 seconds)
        current_time = time.time()
        if current_time - last_predict_time > 2.5:
            # Convert to grayscale to check if the canvas is completely empty before predicting
            gray_canvas = cv2.cvtColor(renderer.canvas, cv2.COLOR_BGR2GRAY)
            if cv2.countNonZero(gray_canvas) > 0:
                canvas_copy = renderer.canvas.copy()
                # Use daemon=True so threads close automatically if you exit the program
                threading.Thread(target=send_canvas_to_model, args=(canvas_copy,), daemon=True).start()
            last_predict_time = current_time

        frame = renderer.blend_and_overlay(frame, drawing_active, current_pos)
        cv2.imshow("Jetson AI Whiteboard", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()