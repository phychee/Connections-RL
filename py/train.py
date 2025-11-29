import torch
import numpy as np
from tqdm import tqdm
from config import GameConfig
from utils import get_device, get_clean_dataframe
from models import MiniLMEmbedding, SimpleGrouper, TransformerEncoderContextualizer, RelationNetworkScorer, ConnectionsDQN
from game import GameState, Game
from experiment import Experiment
from replay import ReplayMemory

def train_agent(
    num_episodes=1000,
    batch_size=32,
    lr=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=None, # Unused, we use dynamic logic
    memory_capacity=10000,
    target_update_freq=100,
    save_path="connections_dqn.pt"
):
    device = get_device()
    print(f"Training on device: {device}")
    
    if epsilon_decay is None:
        decay_target = int(num_episodes * 0.8)
    else:
        decay_target = epsilon_decay
    
    # Load data
    df = get_clean_dataframe()
    
    # Split data: First 600 games for training
    train_game_ids = df["Game ID"].unique()[:600]
    train_df = df[df["Game ID"].isin(train_game_ids)]
    print(f"Training on {len(train_game_ids)} games.")

    # Initialize components
    embedder = MiniLMEmbedding() # Moves to device internally if needed, but here it's CPU based mostly
    # Note: MiniLMEmbedding uses CPU by default in the provided code, but we can move tensors to device later
    
    grouper = SimpleGrouper(device)
    contextualizer = TransformerEncoderContextualizer().to(device)
    scorer = RelationNetworkScorer(device).to(device)
    
    model = ConnectionsDQN(
        embedder=embedder,
        contextualizer=contextualizer,
        grouper=grouper,
        scorer=scorer,
        lr=lr,
        device=device
    ).to(device)
    
    # Target Network
    target_net = ConnectionsDQN(
        embedder=embedder, # Shared embedder (frozen)
        contextualizer=TransformerEncoderContextualizer().to(device), # New instance
        grouper=grouper, # Shared grouper (no params)
        scorer=RelationNetworkScorer(device).to(device), # New instance
        lr=lr,
        device=device
    ).to(device)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()

    memory = ReplayMemory(memory_capacity, device)
    
    experiment = Experiment(
        df=train_df,
        word_cnt=16,
        embedder=embedder,
        grouper=grouper,
        device=device
    )
    
    # Precompute embeddings for training games to save time
    print("Precomputing embeddings...")
    experiment.get_all_word_embeddings()
    # Move embeddings to device
    experiment.embeddings = experiment.embeddings.to(device)

    env = Game()
    config = GameConfig()

    steps_done = 0
    losses = []
    rewards = []
    
    print("Starting training loop...")
    for i_episode in tqdm(range(num_episodes)):
        # Randomly select a game
        game_id = np.random.choice(train_game_ids)
        
        # Setup game
        # We need to efficiently get the board tensors. 
        # The experiment class has a helper, but it might be slow to do it every time if not optimized.
        # For now, we use the helper.
        # board = experiment.df_to_game(game_id)
        # Ensure board tensors are on device
        # board = board # dataclass, fields are tensors.
        # Note: df_to_game calls get_word_embeddings which uses the precomputed ones if we adjusted it, 
        # but currently it slices from self.embeddings if we use all_df_to_game_states logic.
        # Let's adjust how we get the board to be efficient.
        # Actually experiment.df_to_game calls get_word_embeddings which calls embedder.encode.
        # We should use the precomputed embeddings.
        
        # Let's do a quick hack to get embeddings from the big tensor
        # We know each game has 16 words.
        # We need to find the start index of this game_id in the train_df
        # This is a bit tricky without a map.
        # Let's build a map once.
        if not hasattr(experiment, 'game_id_to_idx'):
            experiment.game_id_to_idx = {gid: i*16 for i, gid in enumerate(train_game_ids)}
        
        idx = experiment.game_id_to_idx[game_id]
        game_embeddings = experiment.embeddings[idx:idx+16]
        
        # Reconstruct board
        categories = torch.arange(4, device=device).repeat_interleave(4) # Dummy categories for board structure, actual labels are in dataset
        # Wait, BoardTensors needs group_labels.
        # The dataset has the labels.
        # We need the labels for the game logic (to check if guess is correct).
        # experiment.df_to_game doesn't seem to return labels correctly in the provided code?
        # Let's look at experiment.py... df_to_game uses "categories = torch.arange(4...)" which is WRONG for the game logic.
        # The game logic needs the ACTUAL labels to check correctness.
        # We need to get the actual targets from the DF.
        
        game_df = train_df[train_df["Game ID"] == game_id]
        # Sort by word to match embedding order?
        # The embeddings were generated from df["Word"].tolist().
        # If df is sorted, then yes.
        # In main.py we sorted the df.
        
        # Get targets
        targets = torch.tensor(game_df["Group Level"].values, dtype=torch.long, device=device)
        
        combos = experiment.board_tensors[0].combos if experiment.board_tensors else get_device() # Wait, get_device returns device
        # We need combos.
        from utils import get_all_combos
        combos = get_all_combos(16, 4, device)
        
        from game import BoardTensors
        board = BoardTensors(
            words=game_embeddings,
            group_labels=targets,
            combos=combos
        )

        state = GameState.game_start(board, config)
        
        total_reward = 0
        
        # Linear epsilon decay over 80% of training epochs
        epsilon = epsilon_start - (min(1.0, (i_episode / decay_target)) * (epsilon_start - epsilon_end))
        
        while True:
            steps_done += 1
            
            # Select action
            # We need to pass masked embeddings to the model
            # The model doesn't explicitly handle masking in forward, so we zero out removed words?
            # Or we just pass the state as is and the model sees all words.
            # Ideally we zero out removed words.
            masked_embeddings = board.words.clone()
            masked_embeddings[~state.words_mask] = 0
            
            # Prepare state info
            # Lives: normalize by dividing by 4
            lives = state.lives.float() / 4.0
            # Num groups found: normalize by dividing by 4
            num_groups_found = state.found_groups.sum().float() / 4.0
            
            state_dict = {
                'board': masked_embeddings.unsqueeze(0),
                'lives': lives.unsqueeze(0),
                'num_groups_found': num_groups_found.unsqueeze(0),
                'words_mask': state.words_mask.unsqueeze(0)
            }
            
            # Model forward
            q_values, _ = model(state_dict) # (1, 1820)
            q_values = q_values.squeeze(0)  # (1820,)
            
            action_idx = model.select_action(
                q_values, 
                mask=state.actions_mask, 
                epsilon=epsilon
            )
            
            # Step
            next_state, reward, finished, info = env.make_guess(board, state, action_idx)
            
            total_reward += reward.item()
            
            # Store transition
            # We store masked embeddings
            next_masked_embeddings = board.words.clone()
            next_masked_embeddings[~next_state.words_mask] = 0
            
            next_lives = next_state.lives.float() / 4.0
            next_num_groups_found = next_state.found_groups.sum().float() / 4.0
            
            next_state_dict = {
                'board': next_masked_embeddings,
                'lives': next_lives,
                'num_groups_found': next_num_groups_found,
                'words_mask': next_state.words_mask.unsqueeze(0)
            }
            
            # For memory push, we need unbatched tensors for the current state components
            # But wait, memory.push expects unbatched tensors
            # state_dict above has batched tensors (unsqueeze(0))
            # Let's create unbatched dicts for memory
            
            memory_state_dict = {
                'board': masked_embeddings,
                'lives': lives,
                'num_groups_found': num_groups_found,
                'words_mask': state.words_mask
            }
            
            memory.push(
                memory_state_dict,
                torch.tensor([action_idx], device=device),
                torch.tensor([reward], device=device),
                next_state_dict,
                torch.tensor([finished], device=device, dtype=torch.bool)
            )
            
            state = next_state
            
            # Train
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                loss = model.train_step(batch, target_net, gamma=gamma)
                losses.append(loss)
            
            if finished:
                break
        
        rewards.append(total_reward)
        
        # Update Target Network
        if i_episode % target_update_freq == 0:
            target_net.load_state_dict(model.state_dict())
        
        rewards.append(total_reward)
        
        if (i_episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {i_episode+1}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.2f}")
            
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_agent()
