import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional
from utils import get_all_combos

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, words: list[str]) -> torch.Tensor:
        """
        words: list of N strings
        return: (N, D) tensor
        """
        raise NotImplementedError

class Contextualizer(nn.Module):
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: (B, 16, D)
        return: (B, 16, D) contextualized embeddings
        """
        raise NotImplementedError

class Grouper(nn.Module):
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        emb: (B, 16, D)
        return: (B, 1820, 4, D)
        """
        raise NotImplementedError

class ActionScorer(nn.Module):
    def forward(self, group_embeddings: torch.Tensor) -> torch.Tensor:
        """
        group_embeddings: (B, A, 4, D)
        return: (B, A) or (A,) scores
        """
        raise NotImplementedError

class Model(nn.Module):
    def select_action(self, q_values: torch.Tensor, mask=None) -> int:
        """
        q_values: (A, )
        mask: (A, ) bolean or None
        return: int (choose action index)
        """
        raise NotImplementedError

    def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Passes the word embeddings through the module to get Q-scores
        word_embeddings: (B, 16, D)
        return: (B, 1820): Q-values for all possible actions
        """
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

class MiniLMEmbedding(Embedder):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        super().__init__()
        self.model = SentenceTransformer(model_name, device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, words: list[str]) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.model.encode(
                words,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        return embeddings

    def encode(self, words: list[str]) -> torch.Tensor:
        return self.forward(words)

class GloveEmbedding(Embedder):
    def __init__(self, model_name='glove-wiki-gigaword-100', device='cpu'):
        super().__init__()
        self.device = device
        self.embeddings = {}
        self.vector_size = 100
        
        import os
        import numpy as np
        import urllib.request
        import zipfile
        import shutil
        
        # Project root is parent of the directory containing this file (py/)
        # Assuming models.py is in py/ folder
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        glove_filename = f"{model_name}.txt"
        glove_path = os.path.join(cache_dir, glove_filename)
        
        if not os.path.exists(glove_path):
            print(f"GloVe vectors not found at {glove_path}")
            print(f"Downloading {model_name}...")
            
            # URL for GloVe 6B (which contains 100d)
            url = "https://nlp.stanford.edu/data/glove.6B.zip"
            zip_path = os.path.join(cache_dir, "glove.6B.zip")
            
            try:
                # Download with progress indication would be nice, but simple for now
                with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                
                print("Download complete. Extracting...")
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # The zip contains glove.6B.100d.txt
                    # We map model_name to the file inside zip
                    # For 'glove-wiki-gigaword-100', it corresponds to glove.6B.100d.txt
                    target_file_in_zip = "glove.6B.100d.txt"
                    zip_ref.extract(target_file_in_zip, cache_dir)
                
                # Rename to match expected name
                extracted_path = os.path.join(cache_dir, target_file_in_zip)
                os.rename(extracted_path, glove_path)
                
                # Cleanup
                os.remove(zip_path)
                print(f"GloVe vectors ready at {glove_path}")
                
            except Exception as e:
                print(f"Failed to download/extract GloVe: {e}")
                # Clean up partial files
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                return

        print(f"Loading GloVe from {glove_path}...")
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings[word] = vector
        print("GloVe loaded.")

    def forward(self, words: list[str]) -> torch.Tensor:
        embeddings = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.embeddings:
                vec = self.embeddings[word_lower]
            else:
                # OOV: use zero vector
                vec = np.zeros(self.vector_size)
            embeddings.append(vec)
        
        return torch.tensor(np.array(embeddings), dtype=torch.float32, device=self.device)

    def encode(self, words: list[str]) -> torch.Tensor:
        return self.forward(words)

class TransformerEncoderContextualizer(Contextualizer):

    def __init__(self, d_size=100, n_layers=4, n_head=4, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_size,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False)

    def forward(self, word_embeddings, mask=None):
        """
        Input: (B, 16, 100)
        Returns (B, 16, 100)
        """
        padding_mask = None
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            
            bool_mask = ~mask
            all_padding_rows = bool_mask.all(dim=1)
            if all_padding_rows.any():
                bool_mask[all_padding_rows, 0] = False
            # Make it faster on MPS without nested tensor
            # 0.0 = Keep, -inf = Ignore
            padding_mask = torch.zeros_like(bool_mask, dtype=torch.float)
            padding_mask.masked_fill_(bool_mask, float('-inf'))
        return self.encoder(word_embeddings, src_key_padding_mask=padding_mask)

class SimpleGrouper(Grouper):

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, n:int, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 16, 100)
        Output: (B, 1820, 4, 100)
        """
        combos = get_all_combos(n, 4, self.device)
        # Reshape combos so they are batch-friendly
        # Use advanced indexing: (B, 16, D)[:, (1820, 4)] -> (B, 1820, 4, D)
        return embeddings[:, combos]
    
    def group(self, embeddings: torch.Tensor) -> torch.Tensor:
         return self.forward(embeddings.shape[1], embeddings)

class CosineSimilarityScorer(ActionScorer):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pair_combos = get_all_combos(4, 2, device) # (6, 2)

    def forward(self, group_embeddings, state_info=None):
        """
        group_embeddings: (Batch, A, 4, D)
        state_info: Unused
        returns (B, A) score
        """
        # Optimized implementation to avoid large memory allocation
        # Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        
        w0 = group_embeddings[:, :, 0, :] # (B, A, D)
        w1 = group_embeddings[:, :, 1, :]
        w2 = group_embeddings[:, :, 2, :]
        w3 = group_embeddings[:, :, 3, :]
        
        s01 = nn.functional.cosine_similarity(w0, w1, dim=2, eps=1e-8)
        s02 = nn.functional.cosine_similarity(w0, w2, dim=2, eps=1e-8)
        s03 = nn.functional.cosine_similarity(w0, w3, dim=2, eps=1e-8)
        s12 = nn.functional.cosine_similarity(w1, w2, dim=2, eps=1e-8)
        s13 = nn.functional.cosine_similarity(w1, w3, dim=2, eps=1e-8)
        s23 = nn.functional.cosine_similarity(w2, w3, dim=2, eps=1e-8)
        
        return s01 + s02 + s03 + s12 + s13 + s23

class PolicyNetwork(nn.Module):
    def __init__(self, k: int = 20, state_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        # Input: K scores + state_dim
        self.input_dim = k + state_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k) # Output Q-values for the K actions
        )

    def forward(self, top_k_scores: torch.Tensor, state_info: torch.Tensor) -> torch.Tensor:
        """
        top_k_scores: (B, K)
        state_info: (B, state_dim)
        returns: (B, K)
        """
        combined = torch.cat([top_k_scores, state_info], dim=1)
        return self.net(combined)

class ConnectionsDQN(Model):
    def __init__(self,
                 embedder: Embedder,
                 contextualizer: Contextualizer,
                 grouper: Grouper,
                 scorer: ActionScorer,
                 k: int = 200,
                 lr=1e-4,
                 device='cpu',
                 enable_projection: bool = False):
        super().__init__()
        self.embedder = embedder
        self.contextualizer = contextualizer
        self.grouper = grouper
        self.scorer = scorer
        self.k = k
        self.device = device
        self.enable_projection = enable_projection
        
        # Projection Layer: 100 -> 100
        # This allows the agent to "rotate" embeddings based on game state
        if self.enable_projection:
            self.projection = nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 100)
            ).to(device)
        else:
            self.projection = None
        
        # Policy Network to choose among Top K
        self.policy = PolicyNetwork(k=k, state_dim=2).to(device)
        
        # We only optimize the policy network
        params = list(self.policy.parameters())
        if self.contextualizer is not None:
             params += list(self.contextualizer.parameters())
        # Projection? Maybe not needed if we trust raw embeddings for similarity.
        # But let's keep it trainable if we want to adapt embeddings?
        # User said "fixed method that takes their embeddings as input".
        # So we should probably NOT train projection if we want pure fixed similarity.
        # But if we want to "learn to choose", we might want to adapt embeddings slightly?
        # Let's stick to strict interpretation: Fixed method on embeddings.
        # So we should probably freeze embedder and NOT use projection for scoring?
        # Or use projection but freeze it?
        # Let's keep projection trainable for now as "preprocessing" but the SCORER is fixed.
        if self.projection is not None:
            params += list(self.projection.parameters())

        self.optimizer = optim.Adam(
            params,
            lr=lr
        )

    def select_action(self,
                      q_values: torch.Tensor,
                      top_k: int = 20, # unused
                      mask: Optional[torch.BoolTensor] = None,
                      epsilon: float = 0.0) -> int:
        """
        q_values: (K, )
        mask: (K, ) boolean tensor indicating a valid action among the top K
        """
        if q_values.dim() > 1:
            q_values = q_values.squeeze()
        
        if np.random.random() < epsilon:
            # exploration: choose randomly among (valid ones)
            if mask is not None:
                valid_indices = torch.nonzero(mask).squeeze(1)
                if len(valid_indices) > 0:
                    return int(valid_indices[torch.randint(len(valid_indices), (1,))].item())
                else:
                    return 0
            return int(torch.randint(len(q_values), (1,)).item())
        else:
            # exploitation: choose the best one
            masked_q_values = q_values.clone()
            if mask is not None:
                masked_q_values[~mask] = -float('inf')
            return int(torch.argmax(masked_q_values).item())

    def forward(self, state: dict, force_indices: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state: dict {'board': (B, 16, D), 'lives': (B, 1), 'num_groups_found': (B, 1)}
        returns: 
            q_values: (B, K) - Q-values for the top K groups
            top_k_indices: (B, K) - Original indices (0-1819) of the top K groups
        """
        word_embeddings = state['board']
        lives = state['lives']
        num_groups_found = state['num_groups_found']
        
        # 1. Project
        if self.enable_projection:
            projected_embeddings = self.projection(word_embeddings)
        else:
            projected_embeddings = word_embeddings

        # 2. Group
        grouped_embeddings = self.grouper(16, projected_embeddings) # (B, 1820, 4, D)
        
        # 3. Score all 1820 groups with Fixed Scorer
        # We allow gradients to flow back to projection layer
        all_scores = self.scorer(grouped_embeddings) # (B, 1820)
        
        # Mask invalid actions if mask provided
        if 'actions_mask' in state:
            actions_mask = state['actions_mask'] # (B, 1820)
            all_scores[~actions_mask] = -float('inf')

        # 3. Sort and get Top K
        # We want the indices of the top K scores
        top_k_scores, top_k_indices = torch.topk(all_scores, self.k, dim=1) # (B, K)
        
        # 4. Policy Network
        # Construct state info
        if lives.dim() == 1: lives = lives.unsqueeze(1)
        if num_groups_found.dim() == 1: num_groups_found = num_groups_found.unsqueeze(1)
        state_info = torch.cat([lives, num_groups_found], dim=1) # (B, 2)
        
        # Replace -inf with a safe value (e.g. -10.0) for the network
        # Cosine similarity is [-1, 1], so -10 is safely "very bad"
        safe_top_k_scores = top_k_scores.clone()
        safe_top_k_scores[safe_top_k_scores == -float('inf')] = -10.0
        
        q_values = self.policy(safe_top_k_scores, state_info) # (B, K)
        
        return q_values, top_k_indices, top_k_scores

    def get_best_valid_action(self, state: dict, actions_mask: torch.Tensor) -> int:
        """
        Finds the best valid action index across ALL 1820 groups.
        Used for fallback when Top-K contains no valid moves.
        """
        word_embeddings = state['board']
        
        # 1. Project
        if self.enable_projection:
            projected_embeddings = self.projection(word_embeddings)
        else:
            projected_embeddings = word_embeddings

        # 2. Group
        grouped_embeddings = self.grouper(16, projected_embeddings)
        
        # 3. Score all
        with torch.no_grad():
            all_scores = self.scorer(grouped_embeddings) # (B, 1820)
            
        # Assuming Batch Size = 1 for fallback logic
        scores = all_scores.squeeze(0) # (1820,)
        
        # Mask invalid actions
        # actions_mask is True for valid, False for invalid
        scores[~actions_mask] = -float('inf')
        
        # Get best
        best_idx = torch.argmax(scores).item()
        return int(best_idx)

    def train_step(self, batch, target_net, gamma=0.99):
        state, action, reward, next_state, finished = batch
        
        # Current Q-values
        # action here is the RANK index (0 to K-1)
        q_values, _, _ = self.forward(state)
        current_q_values = q_values.gather(1, action.long())
        
        # Next Q-values
        with torch.no_grad():
            # Double dqn
            next_q_values_online, _, _ = self.forward(next_state)
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
            
            next_q_values_target, _, _ = target_net(next_state)
            next_max_q = next_q_values_target.gather(1, next_actions)
            
            target_q_values = reward + gamma * next_max_q * (~finished)
            
        # Loss
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

class DQN(nn.Module):

    def __init__(self, n_obs, n_actions, n_hidden=128):
        super(DQN, self).__init__()

        self.q = nn.Sequential(
                    nn.Linear(n_obs, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_actions),
                )

    def forward(self, observations):
        return self.q(observations)
