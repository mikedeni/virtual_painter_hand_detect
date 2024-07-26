import os
import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque

# Initialize configuration
webcam = 0
width = 1280
height = 720

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Global variables
countdown_started = False
countdown_time = 0
countdown_value = 0
finger_count_history = deque(maxlen=10)
debounce_frames = 10
cooldown_time = 0
cooldown_duration = 3

# Variables for drawing
drawing = False
last_point = None
current_color = (255, 255, 255)  # Default color is white

# Create a blank canvas with the same size and type as the frame
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Directory to save screenshots
screenshot_dir = "Image"
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# Load the header image
folderPath = "Header"
overlayList = [
    cv2.imread(f"{folderPath}/{imPath}") for imPath in os.listdir(folderPath)
]
header = overlayList[0]


# Function to count extended fingers
def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    return sum(
        1
        for tip in finger_tips
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y
    )


# Countdown function
def start_countdown():
    global countdown_started, countdown_time, countdown_value
    countdown_started = True
    countdown_time = time.time()
    countdown_value = 3


# Function to capture and save a screenshot with canvas blended
def capture_screenshot(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{screenshot_dir}/screenshot_{timestamp}.jpeg"
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved as {filename}")


# Draw line function
def draw_line(frame, start_point, end_point, color=(0, 0, 255), thickness=10):
    cv2.line(frame, start_point, end_point, color, thickness)
    cv2.line(canvas, start_point, end_point, color, thickness)  # Draw on canvas as well


# Function to get landmark positions
def get_landmark_positions(hand_landmarks):
    return {
        id: (int(landmark.x * width), int(landmark.y * height))
        for id, landmark in enumerate(hand_landmarks.landmark)
    }


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip Camera
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[
            0
        ]  # Use the first detected hand only

        # Create custom drawing specs with current color
        landmark_drawing_spec = mp_draw.DrawingSpec(
            color=current_color, thickness=2, circle_radius=2
        )
        connection_drawing_spec = mp_draw.DrawingSpec(color=current_color, thickness=2)

        # Draw hand landmarks with the current color
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec,
            connection_drawing_spec,
        )

        # Count extended fingers
        extended_fingers = count_fingers(hand_landmarks)

        # Update finger count history
        finger_count_history.append(extended_fingers)

        # Track drawing
        if extended_fingers == 1 and drawing:
            # Get the tip of the index finger
            index_tip = hand_landmarks.landmark[8]
            current_point = (
                int(index_tip.x * frame.shape[1]),
                int(index_tip.y * frame.shape[0]),
            )

            if last_point:
                # Draw a line on the canvas and frame
                draw_line(frame, last_point, current_point, color=current_color)

            # Update last_point
            last_point = current_point
        else:
            last_point = None  # Reset the last_point when no finger is extended

        # Check if exactly three fingers are extended steadily for debounce_frames
        if (
            finger_count_history.count(3) == debounce_frames
            and not countdown_started
            and (time.time() - cooldown_time) > cooldown_duration
        ):
            # Start the countdown
            start_countdown()

        if finger_count_history.count(2) == debounce_frames:
            # Get landmark positions
            landmarks_positions = get_landmark_positions(hand_landmarks)
            x1, y1 = landmarks_positions[8]

            if y1 < 140:
                if 300 <= x1 <= 400:
                    # Green Color
                    header = overlayList[1]
                    drawing = True
                    current_color = (0, 255, 0)
                elif 460 <= x1 <= 610:
                    # Red Color
                    header = overlayList[2]
                    drawing = True
                    current_color = (0, 0, 255)
                elif 660 <= x1 <= 850:
                    # Yellow Color
                    header = overlayList[3]
                    drawing = True
                    current_color = (0, 255, 255)
                elif 870 <= x1 <= 1050:
                    # Eraser
                    header = overlayList[4]
                    drawing = True
                    current_color = (0, 0, 0)
                elif 1080 <= x1 <= 1250:
                    # Clear
                    header = overlayList[0]
                    drawing = False
                    current_color = (255, 255, 255)  # default color
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Inverse the canvas for better drawing visibility
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    frame[0:133, 0:1280] = header

    # Capture screenshot only after inverting the canvas
    if countdown_started:
        elapsed_time = time.time() - countdown_time
        if elapsed_time > 1:
            countdown_value -= 1
            countdown_time = time.time()

        if countdown_value > 0:
            # Get text size to center it
            text = f"{countdown_value}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            font_thickness = 5
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_width, text_height = text_size
            text_x = (frame.shape[1] - text_width) // 2
            text_y = (frame.shape[0] + text_height) // 2

            # Put text in the center
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 0),
                font_thickness,
                cv2.LINE_AA,
            )
        else:
            capture_screenshot(frame)
            countdown_started = False
            cooldown_time = time.time()  # Update the cooldown time after screenshot

    # Display the frame with hand detection
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
