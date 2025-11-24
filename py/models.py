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

class Model:
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
        # Assuming embeddings has batch dim B
        B, _, D = embeddings.shape
        combos = combos.unsqueeze(0).expand(B, -1, -1)
        return embeddings.gather(1, combos.unsqueeze(-1).expand(-1, -1, -1, D))
    
    def group(self, embeddings: torch.Tensor) -> torch.Tensor:
         return self.forward(embeddings.shape[1], embeddings)


class RelationNetworkScorer(ActionScorer):

    def __init__(self,
                 device,
                 d_size=384,
                 g_hidden=256,
                 f_hidden=128,
                 output_size=1):
        """
        Inspired by the code here:
        https://diegovianagomes.medium.com/lets-develop-a-simple-neural-network-module-for-relational-reasoning-part-1-97762ca22e01
        """
        super().__init__()

        # This is g_theta from the paper
        self.g = nn.Sequential(
            nn.Linear(2 * d_size, g_hidden),
            nn.ReLU(),
            nn.Linear(g_hidden, g_hidden),
            nn.ReLU()
        )

        # This is f_phi from the paper
        self.f = nn.Sequential(
            nn.Linear(g_hidden, f_hidden),
            nn.ReLU(),
            nn.Linear(f_hidden, output_size)

        )
        self.device = device

        # Get all pairwaise indexes possible of a group of 4
        self.pair_combos = get_all_combos(4, 2, self.device)

    def forward(self, group_embeddings):
        """
        emb: (Batch, 4, 384)
        returns (B,) score
        """
        group_pairs = group_embeddings[:, self.pair_combos]

        # Concatenate the embedding pairs, (B, 6, 2*384)
        group_pairs = group_pairs.reshape(group_embeddings.shape[0],
                                         -1, 2 * group_embeddings.shape[2])

        relations = self.g(group_pairs)

        # Sum over the relations, (B, g_hidden)
        relations_sum = relations.sum(dim=1)

        # Get the scores
        scores = self.f(relations_sum).squeeze(-1)

        return scores

class ConnectionsDQN(Model):
    def __init__(self,
                 embedder: Embedder,
                 contextualizer: Contextualizer,
                 grouper: Grouper,
                 scorer: ActionScorer,
                 lr=1e-4,
                 device='cpu'):
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

    def select_action(self, q_values: torch.Tensor, mask: Optional[torch.BoolTensor] = None, epsilon: float = 0.0) -> int:
        """
        q_values: (A, ) or (1, A) tensor of Q-scores
        mask: (A, ) boolean tensor indicating a valid action
        epsilon: probability of exploration (choosing random action)
        return: int (chosen action index)
        """
        # Input (1820)
        if q_values.dim() > 1:
            print(f"q_values is not 1d, is {q_values.shape}, is this intended?")
            q_values = q_values.squeeze()
        masked_q_values = q_values.clone()
        if mask is not None:
            masked_q_values[~mask] = -float('inf')
        if np.random.random() < epsilon:
            # exploration
            if mask is not None:
                valid_indices = torch.nonzero(mask).flatten()
                if len(valid_indices) > 0:
                    # Pick random idx
                    random_idx = int(torch.randint(len(valid_indices), (1,)).item())
                    return int(valid_indices[random_idx].item())
            # otherwise randomly pick one action
            return int(torch.randint(len(q_values), (1,)).item())
        else:
            # exploitation
            return int(torch.argmax(masked_q_values).item())

    def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Passes the word embeddings through the module to get Q-scores
        word_embeddings: (B, 16, D)
        return: (B, 1820): Q-values for all possible actions
        """
        # if no batch dimension, we make one
        if word_embeddings.dim() == 2:
            # if input: (16,D) -> (1,16,D)=(B,16,D)
            state_emb = word_embeddings.unsqueeze(0)
        # (B, 16, D) -> (B, 16, D)
        contextualized_embeddings = self.contextualizer(word_embeddings)
        # (B, 16, D) -> (B, 1820, 4, D)
        grouped_embeddings = self.grouper(16, contextualized_embeddings) 
        # (B, 1820, 4, D) -> (B, 1820)
        scores = self.scorer(grouped_embeddings)
        return scores

    def train_step(self, batch):
        raise NotImplementedError

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
