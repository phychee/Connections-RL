import torch
import pandas as pd
import kagglehub
from config import GameConfig
from utils import get_all_combos
from dataset import ConnectionsData
from models import MiniLMEmbedding, SimpleGrouper
from game import GameState, Game
from experiment import Experiment

def main():
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Download dataset
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("eric27n/the-new-york-times-connections") + "/" + "Connections_Data.csv"
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = pd.read_csv(dataset_path)
    
    # Remove games where some of the words are NULL
    bad_ids = [59, 62]
    df.drop(df[df["Game ID"].isin(bad_ids)].index, inplace=True)
    
    df = df.sort_values(
        by=["Game ID", "Group Level", "Word"]
    ).reset_index(drop=True)
    
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head())

    # Experiment setup
    print("Setting up experiment...")
    test_exp = Experiment(
        df=df,
        word_cnt=16,
        embedder=MiniLMEmbedding(),
        grouper=SimpleGrouper(device),
        device=device,
    )

    # Test single game
    print("Testing single game setup...")
    test_game = test_exp.df_to_game(1)
    test_game_config = GameConfig()
    test_game_state = GameState.game_start(test_game, test_game_config)
    test_game_env = Game()

    # Make a guess
    print("Making a guess (Action 0)...")
    # Action 0 corresponds to indices [0, 1, 2, 3] which are all Group 0 in the sorted dataframe
    next_state, reward, finished, logs = test_game_env.make_guess(test_game, test_game_state, 0)
    
    print(f"Reward: {reward}")
    print(f"Finished: {finished}")
    print(f"Logs: {logs}")

if __name__ == "__main__":
    main()
