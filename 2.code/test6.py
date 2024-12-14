import cv2
import mediapipe as mp
import math
import pyautogui
from ultralytics import YOLO

# YOLO 모델 로드
yolo_model = YOLO("yolov8n.pt")

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def get_finger_status(hand):
    fingers = []
    if hand.landmark[4].x < hand.landmark[3].x:  # 엄지
        fingers.append(1)
    else:
        fingers.append(0)
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    for tip, pip in zip(tips, pip_joints):
        if hand.landmark[tip].y < hand.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def is_thumb_index_touching(hand):
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.06


def recognize_gesture(fingers_status, thumb_index_touching):
    if fingers_status == [0, 0, 0, 0, 0]:
        return 'fist'
    elif fingers_status == [1, 1, 1, 1, 1]:
        return 'open'
    elif fingers_status == [0, 1, 0, 0, 0]:
        return 'point'
    return 'unknown'


def control_ppt(gesture):
    if gesture == "open":
        print("다음 슬라이드")
        pyautogui.press("right")  # Right Arrow key
    elif gesture == "fist":
        print("이전 슬라이드")
        pyautogui.press("left")  # Left Arrow key


# 웹캠 실행
video = cv2.VideoCapture(0)
print("Webcam is running... Press 'ESC' to exit.")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # YOLO로 손 객체 탐지
    yolo_results = yolo_model(frame)
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] == 0:  # YOLO 클래스 ID 0이 손
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                hand_roi = frame[y1:y2, x1:x2]

                # Mediapipe로 랜드마크 추적
                img_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        fingers_status = get_finger_status(hand_landmarks)
                        thumb_index_touching = is_thumb_index_touching(hand_landmarks)
                        gesture = recognize_gesture(fingers_status, thumb_index_touching)

                        control_ppt(gesture)  # PPT 제어
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture with PPT Control', frame)
    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

video.release()
cv2.destroyAllWindows()