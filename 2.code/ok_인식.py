import cv2
import mediapipe as mp
import math
from ultralytics import YOLO  # YOLO v8 라이브러리 임포트

# YOLO 모델 로드
yolo_model = YOLO("yolov8n.pt")  # YOLO v8 모델

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
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
        fingers.append(1 if hand.landmark[tip].y < hand.landmark[pip].y else 0)
    return fingers


def is_thumb_index_touching(hand):
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.04  # "ok" 제스처 인식 기준


def recognize_gesture(fingers_status, thumb_index_touching):
    if thumb_index_touching and fingers_status == [1, 1, 0, 0, 0]:
        return 'ok'  # 엄지와 검지가 붙어 있는 경우
    if fingers_status == [0, 0, 0, 0, 0]:
        return 'fist'
    elif fingers_status == [0, 1, 0, 0, 0]:
        return 'point'
    elif fingers_status == [1, 1, 1, 1, 1]:
        return 'open'
    elif fingers_status == [0, 1, 1, 0, 0]:
        return 'peace'
    elif fingers_status == [1, 1, 0, 0, 0]:
        return 'standby'
    return 'unknown'


# 웹캠 설정
video = cv2.VideoCapture(0)
print("Webcam is running... Press 'ESC' to exit.")

frame_skip = 2  # YOLO 실행 주기를 줄이기 위한 프레임 스킵 설정
frame_count = 0
hand_boxes = []  # YOLO가 탐지한 손 박스 정보 저장

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # YOLO로 손 탐지 (frame_skip마다 실행)
    if frame_count % frame_skip == 0:
        yolo_results = yolo_model(frame, conf=0.5)
        hand_boxes = []
        for result in yolo_results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # YOLO 클래스 ID 0이 손
                    coords = box.xyxy[0].tolist()
                    hand_boxes.append([int(c) for c in coords])
    frame_count += 1

    # Mediapipe로 손 랜드마크 추적
    if hand_boxes:
        for box in hand_boxes:
            x1, y1, x2, y2 = box
            hand_roi = frame[y1:y2, x1:x2]
            img_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    fingers_status = get_finger_status(hand_landmarks)
                    thumb_index_touching = is_thumb_index_touching(hand_landmarks)
                    gesture = recognize_gesture(fingers_status, thumb_index_touching)
                    print(f"Gesture: {gesture}")

                    # 손 랜드마크와 연결선 그리기
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture with YOLO', frame)
    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

video.release()
cv2.destroyAllWindows()
