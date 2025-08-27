import cv2
import numpy
import mediapipe as mp 
import random
from dataclasses import dataclass
import time
from typing import Optional, Tuple, List
from enum import Enum

# Map for move indices to names
MOVES = {0: "rock", 1: "paper", 2: "scissors"}

# Win conditions mapping - defines when player wins
WIN_MAP = {
    ("rock", "scissors"): "you",
    ("scissors", "paper"): "you",
    ("paper", "rock"): "you",
}

class GameScreen(Enum):
    HOME = "home"
    COUNTDOWN = "countdown"
    REVEAL = "reveal"
    RESULT = "result"


@dataclass
class GameState:
    current_screen: GameScreen = GameScreen.HOME
    countdown_value: int = 3
    countdown_start_time: float = 0.0

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
    elif you == "unknown":
        return "oops didn't quite catch that"
    return "you" if WIN_MAP.get((you, ai)) == "you" else "ai"

def start_countdown(state: GameState):
    state.current_screen = GameScreen.COUNTDOWN
    state.countdown_value = 3
    state.countdown_start_time = time.time()
    state.your_move = None
    state.ai_move = None
    state.result = None

def update_countdown(state: GameState) -> bool:
    elapsed = time.time() - state.countdown_start_time
    state.countdown_value = max(0, 3 - int(elapsed))

    if elapsed >= 3.0:
        if state.current_guess:
            state.your_move = state.current_guess
        else:
            state.your_move = "unknown"
        state.ai_move = ai_move()
        state.result = decide_winner(state.your_move, state.ai_move)

        if state.result == "you":
            state.score_you += 1
        elif state.result == "ai":
            state.score_ai += 1
        
        state.current_screen = GameScreen.REVEAL
        return True
    return False

def go_to_result(state: GameState):
    state.current_screen = GameScreen.RESULT

def go_home(state: GameState):
    state.current_screen = GameScreen.HOME

def reset_scores(state: GameState):
    state.score_you = 0
    state.score_ai = 0

def draw_home_screen(frame, state: GameState):
    h, w = frame.shape[:2]
    
    # Background
    cv2.rectangle(frame, (0, 0), (w, h), (50, 50, 50), -1)
    
    # Title
    title = "ROCK PAPER SCISSORS"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(frame, title, (title_x, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Play button
    button_text = "Click SPACE to Play"
    button_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    button_x = (w - button_size[0]) // 2
    cv2.putText(frame, button_text, (button_x, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Scores
    score_text = f"Score: You {state.score_you} - {state.score_ai} AI"
    score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    score_x = (w - score_size[0]) // 2
    cv2.putText(frame, score_text, (score_x, h//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

def draw_countdown_screen(frame, state: GameState):
    h, w = frame.shape[:2]
    
    # Background
    cv2.rectangle(frame, (0, 0), (w, h), (30, 30, 30), -1)
    
    # Countdown number
    count_text = str(state.countdown_value)
    count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 5)[0]
    count_x = (w - count_size[0]) // 2
    count_y = (h + count_size[1]) // 2
    cv2.putText(frame, count_text, (count_x, count_y), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 5)
    
    # Instructions
    if state.countdown_value > 0:
        instruction = "Get ready!"
    else:
        instruction = "REVEAL!"
    
    inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    inst_x = (w - inst_size[0]) // 2
    cv2.putText(frame, instruction, (inst_x, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

def draw_reveal_screen(frame, state: GameState):
    h, w = frame.shape[:2]
    
    # Background
    cv2.rectangle(frame, (0, 0), (w, h), (40, 40, 40), -1)
    
    # Your move
    your_text = f"Your move: {state.your_move.upper()}"
    your_size = cv2.getTextSize(your_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    your_x = (w - your_size[0]) // 2
    cv2.putText(frame, your_text, (your_x, h//2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # AI move
    ai_text = f"AI move: {state.ai_move.upper()}"
    ai_size = cv2.getTextSize(ai_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    ai_x = (w - ai_size[0]) // 2
    cv2.putText(frame, ai_text, (ai_x, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    
    # Result
    result_text = f"Result: {state.result.upper()}"
    result_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    result_x = (w - result_size[0]) // 2
    cv2.putText(frame, result_text, (result_x, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    
    # Auto-advance instruction
    advance_text = "Press SPACE to continue"
    advance_size = cv2.getTextSize(advance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    advance_x = (w - advance_size[0]) // 2
    cv2.putText(frame, advance_text, (advance_x, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

def draw_result_screen(frame, state: GameState):
    h, w = frame.shape[:2]
    
    # Background
    cv2.rectangle(frame, (0, 0), (w, h), (50, 50, 50), -1)
    
    # Final result
    result_text = f"Result: {state.result.upper()}"
    result_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    result_x = (w - result_size[0]) // 2
    cv2.putText(frame, result_text, (result_x, h//2 - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    
    # Scores
    score_text = f"Score: You {state.score_you} - {state.score_ai} AI"
    score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    score_x = (w - score_size[0]) // 2
    cv2.putText(frame, score_text, (score_x, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    # Buttons
    buttons = [
        ("SPACE - Play Again", (0, 255, 0)),
        ("R - Reset Scores", (255, 165, 0)),
        ("H - Back Home", (255, 255, 0)),
        ("Q - Quit", (255, 100, 0))
    ]
    
    for i, (text, color) in enumerate(buttons):
        y_pos = h//2 + 50 + i * 40
        cv2.putText(frame, text, (w//2 - 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def draw_game_overlay(frame, state: GameState, landmarks: Optional[List[Tuple[int, int]]], temp_guess: Optional[str]):
    """Draw overlay during countdown phase"""
    h, w = frame.shape[:2]
    
    # Background bar
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # Status text
    if temp_guess:
        status = f"Detected: {temp_guess}"
    else:
        status = "Show your hand"
    
    cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw landmarks if present
    if landmarks is not None:
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


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
            h, w = frame.shape[:2]

            # Handle different screens
            if state.current_screen == GameScreen.HOME:
                draw_home_screen(frame, state)
                
            elif state.current_screen == GameScreen.COUNTDOWN:
                # Detect hand during countdown
                landmarks = detect_landmarks(hands, frame)
                temp_guess = None
                if landmarks is not None:
                    fingers = finger_states(landmarks)
                    temp_guess = classify_gesture(fingers)
                
                # Update stabilization
                update_stable_choice(state, temp_guess)
                
                # Draw countdown
                draw_countdown_screen(frame, state)
                draw_game_overlay(frame, state, landmarks, temp_guess)
                
                # Update countdown
                if update_countdown(state):
                    # Countdown complete, wait a moment then show result
                    time.sleep(1.0)
                    go_to_result(state)
                
            elif state.current_screen == GameScreen.REVEAL:
                draw_reveal_screen(frame, state)
                
            elif state.current_screen == GameScreen.RESULT:
                draw_result_screen(frame, state)

            cv2.imshow("RPS AI Game", frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                if state.current_screen == GameScreen.HOME:
                    start_countdown(state)
                elif state.current_screen == GameScreen.REVEAL:
                    go_to_result(state)
                elif state.current_screen == GameScreen.RESULT:
                    start_countdown(state)
            elif key == ord('r') and state.current_screen == GameScreen.RESULT:
                reset_scores(state)
            elif key == ord('h') and state.current_screen == GameScreen.RESULT:
                go_home(state)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()