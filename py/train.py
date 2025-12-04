
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import GameConfig
from utils import get_device, get_clean_dataframe
from models import MiniLMEmbedding, GloveEmbedding, SimpleGrouper, TransformerEncoderContextualizer, CosineSimilarityScorer, ConnectionsDQN
from game import GameState, Game
from experiment import Experiment
from replay import ReplayMemory

def train_agent(
    num_episodes=1000,
    batch_size=64,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=None, # Unused, we use dynamic logic
    memory_capacity=10000,
    target_update_freq=100,
    save_path="model.pt",
    pretrained_path=None
):
    device = get_device()
    print(f"Training on device: {device}")
    print(f"saving to {save_path}")
    
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
    # embedder = MiniLMEmbedding(device=device)
    embedder = GloveEmbedding(device=device)
    # Note: MiniLMEmbedding uses CPU by default in the provided code, but we can move tensors to device later
    
    grouper = SimpleGrouper(device)
    # contextualizer = TransformerEncoderContextualizer().to(device)
    scorer = CosineSimilarityScorer(device).to(device)
    
    model = ConnectionsDQN(
        embedder=embedder,
        contextualizer=None,
        grouper=grouper,
        scorer=scorer,
        k=200,
        lr=lr,
        device=device,
        enable_projection=False
    ).to(device)
    
    if pretrained_path == 'auto':
        from pretrain import pretrain_policy
        pretrain_policy(model, device)
    elif pretrained_path:
        print(f"Loading pretrained model from {pretrained_path}...")
        try:
            # We only want to load the policy network weights if possible, 
            # or the whole state dict if it matches.
            # The pretrain.py saves the whole model.state_dict()
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded.")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
    
    # Target Network
    target_net = ConnectionsDQN(
        embedder=embedder, # Shared embedder (frozen)
        contextualizer=None,
        grouper=grouper, # Shared grouper (no params)
        scorer=CosineSimilarityScorer(device).to(device), # New instance
        k=200,
        lr=lr,
        device=device,
        enable_projection=False
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

    rewards = []
    losses = []
    
    # Metrics for plotting (matching logs)
    log_episodes = []
    log_avg_rewards = []
    log_avg_losses = []
    
    steps_done = 0
    # We need combos.
    from utils import get_all_combos
    static_combos = get_all_combos(16, 4, device)
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
        
        # shuffle data
        perm = torch.randperm(16, device=device)
        game_embeddings = game_embeddings[perm]
        targets = targets[perm]
               
        from game import BoardTensors
        board = BoardTensors(
            words=game_embeddings,
            group_labels=targets,
            combos=static_combos
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

            # Add actions_mask to state_dict for model
            state_dict['actions_mask'] = state.actions_mask.unsqueeze(0)
            
            # Forward
            q_values, top_k_indices, top_k_scores = model(state_dict)
            q_values = q_values.squeeze(0)
            top_k_indices = top_k_indices.squeeze(0)
            top_k_scores = top_k_scores.squeeze(0)
            
            # Determine valid actions in Top-K (those with score > -inf)
            # Since we masked invalid actions with -inf, we just check for that.
            # However, float comparison can be tricky. Let's use a threshold.
            valid_mask = top_k_scores > -1e9
            num_valid = valid_mask.sum().item()
            
            # If no valid actions (should not happen unless K=0 or game over logic fail), break
            if num_valid == 0:
                 # This implies NO valid moves exist in the entire game (since we masked ALL)
                 # This should be handled by game over check, but safety net:
                 break
            
            # Exploration Strategy: Top 1/3 of VALID actions
            if random.random() < epsilon:
                # Limit to top 1/3 of valid actions
                limit = max(1, int(np.ceil(num_valid / 3.0)))
                # Pick random index from 0 to limit-1
                action_idx = random.randint(0, limit - 1)
            else:
                # Greedy: Pick best valid action
                # Since q_values correspond to sorted top_k, index 0 is best scorer-wise?
                # NO! q_values are from Policy Network.
                # Policy Network sees (Score, State).
                # We want argmax Q, but ONLY for valid indices.
                
                # Mask invalid Q-values
                q_values[~valid_mask] = -float('inf')
                action_idx = q_values.argmax().item()

            # Map rank to actual group index
            actual_group_idx = top_k_indices[action_idx].item()
            
            # Step Env
            next_state, reward, finished, info = env.make_guess(board, state, actual_group_idx)
            
            # Update Loop Variables
            total_reward += reward.item()
            state = next_state
            
            # Store transition
            # We need to store actions_mask in memory for training
            memory_state_dict = {
                'board': masked_embeddings,
                'lives': lives,
                'num_groups_found': num_groups_found,
                'words_mask': state.words_mask, # Original mask
                'actions_mask': state.actions_mask # Added
            }
            
            # Next state dict for memory
            next_masked_embeddings = board.words.clone()
            next_masked_embeddings[~next_state.words_mask] = 0
            
            # We need unbatched for memory push?
            # ReplayMemory.push expects tensors or dicts of tensors?
            # It seems it takes whatever we give it.
            # But we need to be consistent with sample().
            
            # Let's clean up next_state_dict for memory
            mem_next_state = {
                'board': next_masked_embeddings,
                'lives': next_state.lives.float() / 4.0, # Unbatched
                'num_groups_found': next_state.found_groups.sum().float() / 4.0, # Unbatched
                'words_mask': next_state.words_mask,
                'actions_mask': next_state.actions_mask
            }
            
            memory.push(
                memory_state_dict,
                torch.tensor([action_idx], device=device),
                torch.tensor([reward], device=device),
                mem_next_state,
                torch.tensor([finished], device=device, dtype=torch.bool)
            )
            
            state = next_state
            
            # Train
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                loss = model.train_step(batch, target_net, gamma=gamma)
                losses.append(loss)
            
            if steps_done % 1000 == 0:
                target_net.load_state_dict(model.state_dict())
            
            if finished:
                break
        
        rewards.append(total_reward)
        
        if (i_episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            
            log_episodes.append(i_episode + 1)
            log_avg_rewards.append(avg_reward)
            log_avg_losses.append(avg_loss)
            
            print(f"Episode {i_episode+1}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.2f}")
            
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Plotting
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(log_episodes, log_avg_rewards)
    plt.title('Average Reward (Past 10 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.savefig('training_rewards.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(log_episodes, log_avg_losses)
    plt.title('Average Loss (Past 100 Steps)')
    plt.xlabel('Episode')
    plt.ylabel('Avg Loss')
    plt.savefig('training_losses.png')
    plt.close()
    
    return model

if __name__ == "__main__":
    train_agent()
