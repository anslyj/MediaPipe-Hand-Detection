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



# makes the ai's choice might change later
def ai_move() -> str:
    return random.choice(list(MOVES.values()))
#this decides the winner
def decide_winner(you: str, ai: str) -> str:
    if you == ai:
        return "draw"
    return "you" if WIN_MAP.get((you, ai)) == "you" else "ai"


def main():
    move = input("What's your move: ")
    ai = ai_move()
    print(ai)
    print (decide_winner(move, ai))

if __name__ == "__main__":
    main()



