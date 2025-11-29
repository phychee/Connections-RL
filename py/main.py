import torch
import pandas as pd
import kagglehub
import argparse
from config import GameConfig
from utils import get_all_combos, get_device, get_clean_dataframe
from dataset import ConnectionsData
from models import MiniLMEmbedding, SimpleGrouper
from game import GameState, Game
from experiment import Experiment
from train import train_agent

def main():
    parser = argparse.ArgumentParser(description="Connections RL Agent")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--play", action="store_true", help="Play/Evaluate the agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--num_games", type=int, default=None, help="Number of games to play/evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()

    if args.train:
        train_agent(
            num_episodes=args.episodes,
            batch_size=args.batch_size
        )
        return

    if args.play:
        from play import evaluate_agent
        evaluate_agent(num_games=args.num_games)
        return

if __name__ == "__main__":
    main()
