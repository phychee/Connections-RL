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
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Connections RL Agent")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--play", action="store_true", help="Play/Evaluate the agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--num_games", type=int, default=None, help="Number of games to play/evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the model")
    args = parser.parse_args()
    
    # save in drive if we're in colab
    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        if args.save_dir is None:
            args.save_dir = "/content/drive/MyDrive/Connections_RL/"
            os.makedirs(args.save_dir, exist_ok=True)
    # If we're running in GCP or locally
    if args.save_dir is None:
        args.save_dir = "./"
    save_path = os.path.join(args.save_dir, "model.pt")
    
    if args.train:
        train_agent(
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            save_path=save_path
        )
        return

    if args.play:
        from play import evaluate_agent
        evaluate_agent(model_path=save_path, num_games=args.num_games)
        return

if __name__ == "__main__":
    main()
