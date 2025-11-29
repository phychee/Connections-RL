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
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_name)
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

class TransformerEncoderContextualizer(Contextualizer):

    def __init__(self, d_size=384, n_layers=4, n_head=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_size,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

    def forward(self, word_embeddings):
        """
        Input: (B, 16, 384)
        Returns (B, 16, 384)
        """
        return self.encoder(word_embeddings)

class SimpleGrouper(Grouper):

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, n:int, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 16, 384)
        Output: (B, 1820, 4, 384)
        """
        combos = get_all_combos(n, 4, self.device)
        # Reshape combos so they are batch-friendly
        # Use advanced indexing: (B, 16, D)[:, (1820, 4)] -> (B, 1820, 4, D)
        return embeddings[:, combos]
    
    def group(self, embeddings: torch.Tensor) -> torch.Tensor:
         return self.forward(embeddings.shape[1], embeddings)

class RelationNetworkScorer(ActionScorer):

    def __init__(self,
                 device,
                 d_model: int = 384,
                 hidden_dim: int = 256,
                 state_dim: int = 2):
        super().__init__()
        self.device = device
        
        # Relation module: takes 2 embeddings (2*D) -> hidden_dim
        self.g = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Scoring module: takes sum of relations (hidden_dim) + state_info (state_dim) -> 1
        self.f = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.pair_combos = get_all_combos(4, 2, device) # (6, 2)

    def forward(self, group_embeddings, state_info):
        """
        group_embeddings: (Batch, A, 4, 384)
        state_info: (Batch, state_dim)
        returns (B, A) score
        """
        B, A, _, D = group_embeddings.shape
        
        flat_embeddings = group_embeddings.view(-1, 4, D) # (B*A, 4, D)
        
        # Create pairs
        group_pairs = flat_embeddings[:, self.pair_combos] # (B*A, 6, 2, D)
        group_pairs = group_pairs.reshape(B * A, 6, 2 * D) # (B*A, 6, 2*D)
        
        # Relational reasoning
        relations = self.g(group_pairs) # (B*A, 6, hidden_dim)
        relations_sum = relations.sum(dim=1) # (B*A, hidden_dim)
        
        # Concatenate state info
        # state_info: (B, state_dim) -> (B, 1, state_dim) -> (B, A, state_dim) -> (B*A, state_dim)
        state_info_expanded = state_info.unsqueeze(1).expand(-1, A, -1).reshape(B * A, -1)
        
        combined = torch.cat([relations_sum, state_info_expanded], dim=1) # (B*A, hidden_dim + state_dim)

        # Get the scores
        scores = self.f(combined) # (B*A, 1)
        scores = scores.view(B, A) # (B, A)

        return scores

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
                 k: int = 1820, # unused, kept for compatibility
                 lr=1e-4,
                 device='cpu'):
        super().__init__()
        self.device = device

        # TODO: better to precompute all word embeddings and to only map: word -> embedding here
        self.embedder = embedder
        self.contextualizer = contextualizer
        self.grouper = grouper
        self.scorer = scorer
        
        self.optimizer = optim.Adam(
            list(self.contextualizer.parameters()) + list(self.scorer.parameters()),
            lr=lr
        )

    def select_action(self,
                      q_values: torch.Tensor,
                      top_k: int = 1820, # unused
                      mask: Optional[torch.BoolTensor] = None,
                      epsilon: float = 0.0) -> int:
        """
        q_values: (1820, ) or (1, 1820) tensor of Q-scores for the TOP K actions
        mask: (1820, ) boolean tensor indicating a valid action among the top K
              Wait, the mask passed from outside is usually (1820,).
              We need to handle masking differently now.
              The mask should be applied BEFORE top-k if possible, or we need to map the mask to top-k.
              
              Actually, for simplicity in this refactor step, let's assume the policy selects among the top K,
              and we handle validity checks in the environment (illegal move penalty).
              Or we can pass the mask for the top K indices.
        """
        # Input
        if q_values.dim() > 1:
            q_values = q_values.squeeze()
        
        masked_q_values = q_values.clone()
        if mask is not None:
            masked_q_values[~mask] = -float('inf')
            
        if np.random.random() < epsilon:
            # exploration: choose randomly among (valid ones)
            if mask is not None:
                valid_indices = torch.nonzero(mask).squeeze(1)
                if len(valid_indices) > 0:
                    return int(valid_indices[torch.randint(len(valid_indices), (1,))].item())
            return int(torch.randint(len(q_values), (1,)).item())
        else:
            # exploitation: choose the best one
            return int(torch.argmax(masked_q_values).item())

    def forward(self, state: dict, force_indices: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passes the word embeddings through the module to get Q-scores
        state: dict {'board': (B, 16, D), 'lives': (B, 1), 'num_groups_found': (B, 1)}
        force_indices: (B, 1) - Optional indices to force into the Top K (for training)
        returns: 
            q_values: (B, K) - Q-values for the top K groups
            top_k_indices: (B, K) - Original indices (0-1819) of the top K groups
        """
        word_embeddings = state['board']
        lives = state['lives']
        num_groups_found = state['num_groups_found']

        # Contextualize
        context_embeddings = self.contextualizer(word_embeddings)
        
        # Group
        grouped_embeddings = self.grouper(16, context_embeddings) # (B, 1820, 4, D)
        
        # Construct state info
        # lives: (B,) -> (B, 1)
        # num_groups_found: (B,) -> (B, 1)
        if lives.dim() == 1:
            lives = lives.unsqueeze(1)
        if num_groups_found.dim() == 1:
            num_groups_found = num_groups_found.unsqueeze(1)
            
        state_info = torch.cat([lives, num_groups_found], dim=1) # (B, 2)
        
        # Score all 1820 groups
        all_scores = self.scorer(grouped_embeddings, state_info) # (B, 1820)
        
        return all_scores, None

    def train_step(self, batch, target_net, gamma=0.99):
        state, action, reward, next_state, finished = batch
        
        # Current Q-values
        q_values, _ = self.forward(state)
        current_q_values = q_values.gather(1, action.long())
        
        # Next Q-values
        with torch.no_grad():
            # Use target_net for next state Q-values
            # We don't force indices here, we just want the best of Top K
            next_q_values, _ = target_net(next_state)
            # Max next Q-value among the Top K
            next_max_q, _ = next_q_values.max(dim=1, keepdim=True)
            
            target_q_values = reward + gamma * next_max_q * (~finished)
            
        # Loss
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
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
