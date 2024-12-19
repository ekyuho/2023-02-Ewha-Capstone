import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 손 제스처 태깅
def detect_rock_paper_scissors(hand_landmarks):
    if hand_landmarks:
        thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks[mp_hands.HandLandmark.PINKY_TIP]

        # "V" 제스처 (Scissors)
        if (
            index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and
            abs(index_tip.x - middle_tip.x) > 0.1 and  
            ring_tip.y > middle_tip.y and pinky_tip.y > middle_tip.y
        ):
            return "Scissors"
        # 바위 (Rock)
        elif (
            all(finger.y > thumb_tip.y for finger in [index_tip, middle_tip, ring_tip, pinky_tip])
        ):
            return "Rock"
        # 보 (Paper)
        elif (
            all(finger.y < thumb_tip.y for finger in [index_tip, middle_tip, ring_tip, pinky_tip])
        ):
            return "Paper"
    return None

# 요가 포즈 태깅
def detect_yoga_pose(pose_landmarks):
    if pose_landmarks:
        left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # 양쪽 어깨가 수평이고 엉덩이보다 위에 있는 경우
        if abs(left_shoulder.y - right_shoulder.y) < 0.05 and \
           left_shoulder.y < left_hip.y and right_shoulder.y < right_hip.y:
            return "Mountain Pose"
        # 한쪽 어깨가 다른 쪽보다 높고 엉덩이 위에 있는 경우
        elif left_shoulder.y < left_hip.y and right_shoulder.y > right_hip.y:
            return "Tree Pose"
        else:
            return "Unknown Pose"
    return None

# 프레임 간 결과 안정화
gesture_queue = deque(maxlen=5)
pose_queue = deque(maxlen=5)

# 비디오 캡처
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands, \
     mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # MediaPipe 모델 실행
        hand_results = hands.process(image)
        pose_results = pose.process(image)

        # 원래 색상으로 복원
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gesture, pose_tag = None, None

        # 손 제스처 인식
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_rock_paper_scissors(hand_landmarks.landmark)

        # 요가 포즈 인식
        if pose_results.pose_landmarks:
            # 얼굴 랜드마크 제외 (0~10번)
            pose_landmarks = np.array([[lm.x, lm.y, lm.z] for i, lm in enumerate(pose_results.pose_landmarks.landmark) if i > 10])
            for landmark in pose_landmarks:
                x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            pose_tag = detect_yoga_pose(pose_results.pose_landmarks.landmark)

        # 프레임 안정화
        if gesture:
            gesture_queue.append(gesture)
        if pose_tag:
            pose_queue.append(pose_tag)

        # 가장 빈도가 높은 결과 선택
        gesture = max(set(gesture_queue), key=gesture_queue.count) if gesture_queue else None
        pose_tag = max(set(pose_queue), key=pose_queue.count) if pose_queue else None

        # 인식된 제스처 또는 포즈 출력
        if gesture:
            cv2.putText(image, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif pose_tag:
            cv2.putText(image, f"Pose: {pose_tag}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 결과 출력
        cv2.imshow('MediaPipe Pose & Hands', image)

        # 종료 키
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
