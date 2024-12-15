import cv2
import mediapipe as mp
import math
import time  # 시간 지연을 위한 모듈
import pyautogui  # 키보드 입력 시뮬레이션을 위한 라이브러리
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
    return distance < 0.03  # "ok" 제스처 인식 기준 (더 엄격하게 조정)

def recognize_gesture(fingers_status, thumb_index_touching):
    if fingers_status == [1, 1, 0, 0, 1]:  # 엄지, 검지, 소지만 펴진 경우
        return 'rock and roll'
    if fingers_status == [0, 0, 0, 0, 0]:
        return 'fist'
    if thumb_index_touching:  # 엄지와 검지만 접촉하고 나머지는 접힘
        return 'ok'
    
    return 'unknown'

# 웹캠 설정
video = cv2.VideoCapture(0)
print("Webcam is running... Press 'ESC' to exit.")

frame_skip = 2  # YOLO 실행 주기를 줄이기 위한 프레임 스킵 설정
frame_count = 0
hand_boxes = []  # YOLO가 탐지한 손 박스 정보 저장

# 슬라이드 이동 제한 시간 설정
last_slide_time = 0  # 마지막으로 슬라이드를 넘긴 시간
slide_delay = 1.5  # 슬라이드 딜레이 (초)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape  # 프레임 크기 가져오기

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

            # Mediapipe에 전달하기 위한 ROI 비율 변환
            img_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # ROI 크기를 전체 프레임 좌표로 변환
                    for landmark in hand_landmarks.landmark:
                        landmark.x = (x1 + landmark.x * (x2 - x1)) / w
                        landmark.y = (y1 + landmark.y * (y2 - y1)) / h

                    fingers_status = get_finger_status(hand_landmarks)
                    thumb_index_touching = is_thumb_index_touching(hand_landmarks)
                    gesture = recognize_gesture(fingers_status, thumb_index_touching)
                    #print(f"Gesture: {gesture}")

                    # OK 제스처가 감지되면 슬라이드 이동
                    current_time = time.time()
                    if gesture == 'ok' and (current_time - last_slide_time) > slide_delay:
                        print("Moving to the next slide...")
                        pyautogui.press('right')  # 오른쪽 화살표 키 입력
                        last_slide_time = current_time  # 슬라이드 이동 시간 갱신

                    # ROCK AND ROLL 제스처가 감지되면 슬라이드 이전 이동
                    if gesture == 'rock and roll' and (current_time - last_slide_time) > slide_delay:
                        print("Moving to the previous slide...")
                        pyautogui.press('left')  # 왼쪽 화살표 키 입력
                        last_slide_time = current_time  # 슬라이드 이동 시간 갱신

                    # 손 랜드마크와 연결선 그리기
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )

    cv2.imshow('Hand Gesture with YOLO', frame)
    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

video.release()
cv2.destroyAllWindows()
