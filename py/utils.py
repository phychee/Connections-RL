import torch
import pandas as pd
import kagglehub

def get_all_combos(n: int, k:int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    # this returns combinations in sorted order
    return torch.combinations(idx, r=k)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def get_clean_dataframe() -> pd.DataFrame:
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("eric27n/the-new-york-times-connections") + "/" + "Connections_Data.csv"
    
    print("Loading and cleaning data...")
    df = pd.read_csv(dataset_path)
    
    # Remove games where some of the words are NULL
    bad_ids = [59, 62]
    df.drop(df[df["Game ID"].isin(bad_ids)].index, inplace=True)
    
    df = df.sort_values(
        by=["Game ID", "Group Level", "Word"]
    ).reset_index(drop=True)
    
    return df

def inspect_data():
    from config import GameConfig
    from models import MiniLMEmbedding, SimpleGrouper
    from game import GameState, Game
    from experiment import Experiment
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    # Load and clean data
    df = get_clean_dataframe()
    
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
