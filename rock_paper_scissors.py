import cv2
import numpy
import mediapipe as mp 
import random


MOVES = {0: "rock", 1: "paper", 2: "scissors"}
WIN_MAP = {
    ("rock", "scissors"): "you",
    ("scissors", "paper"): "you",
    ("paper", "rock"): "you",
}

def ai_move() -> str:
    return random.choice(list(MOVES.values()))

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



