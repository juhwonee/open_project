import cv2
import mediapipe as mp
import math

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def get_finger_status(hand):
    """
    손가락이 펴져 있는지 접혀 있는지 확인하는 함수
    """
    fingers = []

    # 엄지: 랜드마크 4가 랜드마크 2의 오른쪽에 있으면 펼쳐진 상태
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
    return distance < 0.06  # 붙어 있는 거리 기준 살짝 완화


def recognize_gesture(fingers_status, thumb_index_touching):
    """
    손 동작을 인식하여 제스처 이름 반환
    """
    print(f"Finger status: {fingers_status}, Thumb-Index touching: {thumb_index_touching}")

    # 엄지와 검지가 붙고 나머지 손가락이 접힌 경우 'three'
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

print("Webcam is running... Press 'ESC' to exit.")
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingers_status = get_finger_status(hand_landmarks)
            thumb_index_touching = is_thumb_index_touching(hand_landmarks)
            gesture = recognize_gesture(fingers_status, thumb_index_touching)
            print(gesture)

            # 손 랜드마크와 연결선 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture', frame)
    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

video.release()
cv2.destroyAllWindows()
