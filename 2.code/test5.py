import cv2
import mediapipe as mp
import math
from ultralytics import YOLO  # YOLO v8 라이브러리 임포트

# YOLO 모델 로드
yolo_model = YOLO("yolov8n.pt")  # YOLO v8 모델. 필요 시 custom 모델을 사용하세요.

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def get_finger_status(hand):
    """
    손가락이 펴져 있는지 접혀 있는지 확인하는 함수
    """
    fingers = []

    # 엄지: 랜드마크 4가 랜드마크 3의 오른쪽에 있으면 펼쳐진 상태
    if hand.landmark[4].x < hand.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 나머지 손가락: 각 손가락의 팁 (8, 12, 16, 20)이 PIP (6, 10, 14, 18) 위에 있으면 펼쳐진 상태
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    for tip, pip in zip(tips, pip_joints):
        if hand.landmark[tip].y < hand.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def is_thumb_index_touching(hand):
    """
    엄지와 검지가 붙어 있는지 확인하는 함수
    """
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.06  # 붙어 있는 거리 기준


def recognize_gesture(fingers_status, thumb_index_touching):
    """
    손 동작을 인식하여 제스처 이름 반환
    """
    print(f"Finger status: {fingers_status}, Thumb-Index touching: {thumb_index_touching}")

    if thumb_index_touching and fingers_status == [1, 1, 0, 0, 0]:
        return 'three'
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
frame_count = 0

print("Webcam is running... Press 'ESC' to exit.")
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # YOLO를 사용해 손 객체 탐지
    yolo_results = yolo_model(frame)
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            # YOLO의 탐지 결과 중 'hand' 클래스만 Mediapipe에 전달
            if box.cls[0] == 0:  # YOLO 클래스 ID 0이 손이라고 가정
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 탐지된 손 좌표
                hand_roi = frame[y1:y2, x1:x2]

                # Mediapipe로 손 랜드마크 추적
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

                        # 손가락이 모두 구부러졌을 때 캡처 저장
                        if fingers_status == [0, 0, 0, 0, 0]:
                            frame_count += 1
                            filename = f"captured_frame_{frame_count}.png"
                            cv2.imwrite(filename, frame)
                            print(f"Captured frame saved as {filename}")

    cv2.imshow('Hand Gesture with YOLO', frame)
    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

video.release()
cv2.destroyAllWindows()