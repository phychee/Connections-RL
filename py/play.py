import torch
import numpy as np
from tqdm import tqdm
from utils import get_device, get_clean_dataframe, get_all_combos
from dataset import ConnectionsData
from models import ConnectionsDQN, MiniLMEmbedding, SimpleGrouper, TransformerEncoderContextualizer, RelationNetworkScorer
from game import Game, GameConfig, GameState, BoardTensors
from experiment import Experiment

def evaluate_agent(model_path="connections_dqn.pt", num_games=None):
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    df = get_clean_dataframe()
    
    # Filter for test games (ID > 600)
    test_df = df[df["Game ID"] > 600]
    test_game_ids = test_df["Game ID"].unique()
    
    if num_games:
        test_game_ids = test_game_ids[:num_games]
        
    print(f"Evaluating on {len(test_game_ids)} games.")
    
    # Initialize components
    embedder = MiniLMEmbedding()
    grouper = SimpleGrouper(device)
    # We need to initialize the model structure to load weights
    # Note: We must match the architecture used in train.py
    # In train.py:
    # contextualizer = TransformerEncoderContextualizer(d_model=384, nhead=4, num_layers=2).to(device)
    # scorer = RelationNetworkScorer(device, d_model=384, hidden_dim=256).to(device)
    # model = ConnectionsDQN(...)
    
    # Let's assume default parameters for now, but ideally these should be in config
    contextualizer = TransformerEncoderContextualizer().to(device)
    scorer = RelationNetworkScorer(device).to(device)
    
    model = ConnectionsDQN(
        embedder=embedder,
        contextualizer=contextualizer,
        grouper=grouper,
        scorer=scorer,
        device=device
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train first.")
        return

    # Experiment setup for embeddings
    experiment = Experiment(
        df=test_df, # Use test_df here? Or full df? 
        # If we use test_df, get_all_word_embeddings might re-encode.
        # Ideally we use the same experiment setup as training to cache? 
        # But for evaluation we can just re-instantiate.
        word_cnt=16,
        embedder=embedder,
        grouper=grouper,
        device=device
    )
    
    # Precompute embeddings for test games
    print("Precomputing embeddings for test set...")
    experiment.get_all_word_embeddings()
    experiment.embeddings = experiment.embeddings.to(device)
    
    # Map game_id to index in test_df
    # Note: experiment.embeddings corresponds to the unique words in the df passed to it.
    # If we passed test_df, index 0 is the first word of the first game in test_df.
    game_id_to_idx = {gid: i*16 for i, gid in enumerate(test_game_ids)}

    env = Game()
    config = GameConfig()
    
    total_rewards = []
    wins = 0
    
    for game_id in tqdm(test_game_ids):
        # Setup board
        idx = game_id_to_idx[game_id]
        game_embeddings = experiment.embeddings[idx:idx+16]
        
        game_df_subset = test_df[test_df["Game ID"] == game_id]
        targets = torch.tensor(game_df_subset["Group Level"].values, dtype=torch.long, device=device)
        combos = get_all_combos(16, 4, device)
        
        board = BoardTensors(
            words=game_embeddings,
            group_labels=targets,
            combos=combos
        )
        
        state = GameState.game_start(board, config)
        total_reward = 0
        
        while True:
            # Select action (Greedy)
            masked_embeddings = board.words.clone()
            masked_embeddings[~state.words_mask] = 0
            
            # Prepare state info
            lives = state.lives.float() / 4.0
            num_groups_found = state.found_groups.sum().float() / 4.0
            
            with torch.no_grad():
                q_values = model(masked_embeddings.unsqueeze(0), lives.unsqueeze(0), num_groups_found.unsqueeze(0)).squeeze(0)
                # Top-k logic is inside select_action, but with epsilon=0 it picks the best of top_k
                action_idx = model.select_action(q_values, top_k=20, mask=state.actions_mask, epsilon=0.0)
            
            # Step
            next_state, reward, finished, info = env.make_guess(board, state, action_idx)
            
            total_reward += reward.item()
            state = next_state
            
            if finished:
                # Check if won
                if state.found_groups.all():
                    wins += 1
                break
        
        total_rewards.append(total_reward)
        
    avg_reward = np.mean(total_rewards)
    win_rate = wins / len(test_game_ids)
    
    print("\nEvaluation Results:")
    print(f"Total Games Played: {len(test_game_ids)}")
    print(f"Games Won: {wins}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Reward: {avg_reward:.4f}")
