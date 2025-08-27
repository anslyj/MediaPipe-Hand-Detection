import cv2
import numpy
import mediapipe as mp 
import random
from dataclasses import dataclass
import time
from typing import Optional, Tuple, List

# Map for move indices to names
MOVES = {0: "rock", 1: "paper", 2: "scissors"}

# Win conditions mapping - defines when player wins
WIN_MAP = {
    ("rock", "scissors"): "you",
    ("scissors", "paper"): "you",
    ("paper", "rock"): "you",
}


@dataclass
class GameState:
    """Tracks the current game state and scores"""
    score_you: int = 0
    score_ai: int = 0
    last_locked_gesture: Optional[str] = None
    last_ai_move: Optional[str] = None
    last_result: Optional[str] = None
    
    # Gesture stabilization parameters
    current_guess: Optional[str] = None
    stable_since_ms: float = 0.0
    lock_ms: int = 700  # Time to hold gesture before locking (may adjust later)


#mediapipe 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def detect_landmarks(hands, frame_bgr) -> Optional[List[Tuple[int, int]]]:
    # Returns the first hand's 21 (x, y) pixel landmarks or None
    
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    # For simplicity: use the first detected hand
    handLms = res.multi_hand_landmarks[0]
    pts = [(int(p.x * w), int(p.y * h)) for p in handLms.landmark]
    return pts


def finger_states(landmarks: List[Tuple[int, int]]) -> List[bool]:
    # Landmark indices for fingertips and joint positions
    tips = [4, 8, 12, 16, 20]  # Fingertip landmarks
    pips = [3, 6, 10, 14, 18]  # PIP joint landmarks
    
    # Special handling for thumb (horizontal extension)
    thumb_tip, thumb_ip, index_mcp = landmarks[4], landmarks[3], landmarks[5]
    thumb_extended = abs(thumb_tip[0] - index_mcp[0]) > abs(thumb_ip[0] - index_mcp[0]) + 8
    
    # Check other fingers (vertical extension - tip above pip joint)
    fingers = [thumb_extended]
    for tip, pip in zip(tips[1:], pips[1:]):
        fingers.append(landmarks[pip][1] - landmarks[tip][1] > 15)  # tip higher than pip
    
    return fingers

def classify_gesture(fingers: List[bool]) -> str:
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

def update_stable_choice(state: GameState, guess: Optional[str]) -> Optional[str]:
    """
    Returns locked gesture if stabilized long enough; otherwise None.
    - Keep the same guess for lock_ms before returning it as locked.
    """
    now_ms = time.time() * 1000
    if guess is None:
        state.current_guess = None
        state.stable_since_ms = 0
        return None

    if guess != state.current_guess:
        state.current_guess = guess
        state.stable_since_ms = now_ms
        return None

    if now_ms - state.stable_since_ms >= state.lock_ms:
        return state.current_guess

    return None

def draw_overlay(frame, state: GameState, landmarks: Optional[List[Tuple[int, int]]], temp_guess: Optional[str]):
    h, w = frame.shape[:2]
    # Background bar
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    # Status text
    status = "Show a gesture"
    if temp_guess:
        status = f"Holding: {temp_guess}"
    if state.last_locked_gesture:
        status = f"Locked: {state.last_locked_gesture} | AI: {state.last_ai_move} | Result: {state.last_result}"

    cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"You {state.score_you} : {state.score_ai} AI", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 255, 180), 2)

    # Draw landmarks if present
    if landmarks is not None:
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

def draw_help(frame):
    h, w = frame.shape[:2]
    help_lines = [
        "Controls: q=quit, r=reset scores",
        "Tips: center your hand, good lighting",
    ]
    y = h - 10
    for line in reversed(help_lines):
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        y -= 22


# makes the ai's choice might change later
def ai_move() -> str:
    return random.choice(list(MOVES.values()))
#this decides the winner
def decide_winner(you: str, ai: str) -> str:
    if you == ai:
        return "draw"
    return "you" if WIN_MAP.get((you, ai)) == "you" else "ai"

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
                state.last_ai_move = ai_move()
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


