import cv2
import numpy
import mediapipe as mp 
import random
from dataclasses import dataclass
import time
from typing import Optional, Tuple, List


MOVES = {0: "rock", 1: "paper", 2: "scissors"}
#helps decide the winner
WIN_MAP = {
    ("rock", "scissors"): "you",
    ("scissors", "paper"): "you",
    ("paper", "rock"): "you",
}

@dataclass
class GameState:
    user_score: int = 0
    ai_score: int = 0
    stable_since_ms: float = 0.0
    lock_ms: int = 700 #may have to tune

#mediapipe 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def finger_states(landmarks: List[Tuple[int, int]]) -> List[bool]:
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    thumb_tip, thumb_ip, index_mcp = landmarks[4], landmarks[3], landmarks[5]
    thumb_extended = abs(thumb_tip[0] - index_mcp[0]) > abs(thumb_ip[0] - index_mcp[0]) + 8

    fingers = [thumb_extended]
    for tip, pip in zip(tips[1:], pips[1:]):
        fingers.append(landmarks[pip][1] - landmarks[tip][1] > 15)  # tip higher than pip
    return fingers

def classifying_gesture(fingers: List[bool]) -> str:
    thumb, idx, mid, ring, pink = fingers
    count = sum(fingers)

    if count <= 1:
        return "rock"
    if idx and mid and not ring and not pink:
        return "scissors"
    if idx and mid and ring and pink:
        return "paper"
    if count == 5:
        return "paper"
    if idx and mid:
        return "scissors"
    return "rock"



# makes the ai's choice might change later
def ai_move() -> str:
    return random.choice(list(MOVES.values()))
#this decides the winner
def decide_winner(you: str, ai: str) -> str:
    if you == ai:
        return "draw"
    return "you" if WIN_MAP.get((you, ai)) == "you" else "ai"

def detect_landmarks(hands, frame_bgr) -> Optional[List[Tuple[int, int]]]:
    # Returns the first hand's 21 (x, y) pixel landmarks or None
    frame = cv2.flip(frame_bgr, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    # For simplicity: use the first detected hand
    handLms = res.multi_hand_landmarks[0]
    pts = [(int(p.x * w), int(p.y * h)) for p in handLms.landmark]
    return pts


def main():
    cap = cv2.VideoCapture(0)
    state = GameState()

    with mp_hands.Hands(model_complexity=0, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            landmarks = detect_landmarks(hands, frame)
            temp_guess = None

            if landmarks is not None:
                fingers = finger_states(landmarks)
                temp_guess = classify_gesture(fingers)

            locked = update_stable_choice(state, temp_guess)

            if locked is not None:
                state.last_locked_gesture = locked
                state.last_ai_move = ai_move_simple()
                state.last_result = decide_winner(state.last_locked_gesture, state.last_ai_move)
                if state.last_result == "you":
                    state.score_you += 1
                elif state.last_result == "ai":
                    state.score_ai += 1
                # Reset stability to avoid double triggers
                state.current_guess = None
                state.stable_since_ms = 0

            draw_overlay(frame, state, landmarks, temp_guess)
            draw_help(frame)

            cv2.imshow("RPS AI", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                state = GameState()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


