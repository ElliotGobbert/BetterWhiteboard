import cv2
import time
import threading
import requests
from camera_tracker import HandTracker
from gesture_recognizer import is_hand_closed, count_extended_fingers
from canvas_renderer import CanvasRenderer

# Locks to prevent thread pileups if the server is running slow
frame_lock = threading.Lock()
is_sending_frame = False

def push_frame_to_server(frame):
    """Streams the live video to the web UI."""
    global is_sending_frame
    _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    try:
        requests.post("http://localhost:8000/update_frame", files={"frame": ("frame.jpg", encoded.tobytes(), "image/jpeg")})
    except requests.exceptions.RequestException:
        pass
    finally:
        with frame_lock:
            is_sending_frame = False

def send_canvas_to_model(canvas):
    """Sends the drawing to the AI for guessing."""
    _, encoded_image = cv2.imencode('.jpg', canvas)
    try:
        requests.post("http://localhost:8000/predict", files={"file": ("canvas.jpg", encoded_image.tobytes(), "image/jpeg")})
    except requests.exceptions.RequestException:
        pass

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(max_hands=2)
    renderer = CanvasRenderer(1280, 720)

    prev_pos = None
    left_prev_x = None  
    last_predict_time = time.time()
    
    global is_sending_frame

    while cap.isOpened():
        success, raw_frame = cap.read()
        if not success:
            continue

        raw_frame = cv2.flip(raw_frame, 1)
        h, w, _ = raw_frame.shape
        
        results = tracker.process_frame(raw_frame)

        drawing_active = False
        current_pos = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label 

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
                        color, thick = (0, 0, 0), 60 
                    else:
                        drawing_active = False
                        prev_pos = None

                    if drawing_active:
                        if prev_pos is None: prev_pos = current_pos
                        renderer.draw_line(prev_pos, current_pos, color, thick)
                        prev_pos = current_pos
                        
                elif hand_label == "Left":
                    fingers_up = count_extended_fingers(hand_landmarks)
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    
                    if fingers_up >= 4:
                        if left_prev_x is not None:
                            velocity = wrist_x - left_prev_x
                            if abs(velocity) > 80:
                                renderer.clear()
                                left_prev_x = None 
                                continue
                        left_prev_x = wrist_x
                    else:
                        left_prev_x = None
        else:
            prev_pos = None
            left_prev_x = None

        # Blend the drawing onto the camera feed
        blended_frame = renderer.blend_and_overlay(raw_frame, drawing_active, current_pos)

        # 1. Background task: Stream the video feed to the website continuously
        with frame_lock:
            if not is_sending_frame:
                is_sending_frame = True
                threading.Thread(target=push_frame_to_server, args=(blended_frame.copy(),), daemon=True).start()

        # 2. Background task: Send canvas for AI prediction every 2.5 seconds
        current_time = time.time()
        if current_time - last_predict_time > 2.5:
            gray_canvas = cv2.cvtColor(renderer.canvas, cv2.COLOR_BGR2GRAY)
            if cv2.countNonZero(gray_canvas) > 0:
                threading.Thread(target=send_canvas_to_model, args=(renderer.canvas.copy(),), daemon=True).start()
            last_predict_time = current_time

        cv2.imshow("Jetson AI Whiteboard", blended_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()