import torch
import torch.nn as nn
from models import Embedder, Grouper
from game import BoardTensors
from utils import get_all_combos

class Experiment:

    # Set these params to None initially for initial validation of the class, remember to remove these defaults later
    # Also, not sure we need env yet.
    def __init__(self,
                 df,
                 word_cnt,
                 embedder: Embedder,
                 device,
                 batch_size=16,
                 grouper: Grouper | None = None,
                 contextualizer: nn.Module | None = None,
                 scorer: nn.Module | None = None,
                 agent=None,
                 env=None,
                 embeddings=None):
        self.df = df
        self.n = word_cnt
        self.embedder = embedder
        self.grouper = grouper
        self.device = device
        self.B = batch_size
        self.contextualizer = contextualizer
        self.scorer = scorer
        self.agent = agent
        self.env = env
        self.embeddings = None
        self.board_tensors = []

    def get_all_word_embeddings(self):
        """
        1. Step
        Helper to get all word embeddings
        Returns: (num_games * 16, 384)
        """
        if not self.embeddings:
            embeddings = self.embedder.encode(self.df["Word"].tolist())
            torch.save(embeddings, f"{self.embeddings.__class__.__name__}_embeddings.pt")
            self.embeddings = embeddings

    def _contextualize(self, batched_embs: torch.Tensor) -> torch.Tensor:
        """
        2nd Step
        For each group of 16 words contextualize the embeddings
        Returns: (B, 16, 384)
        """
        outputs = []
        for batch in batched_embs:
            ctx_embeddings = self.contextualizer.forward(batch)
            outputs.append(ctx_embeddings)
        return ctx_embeddings

    def _group_embeddings(self, batched_embs: torch.Tensor) -> torch.Tensor:
        """
        3rd Step
        For each board we output all possible groups of words
        Returns: (B, 1820, 4, 384)
        """
        outputs = []
        for batch in batched_embs:
            # Note: The notebook called self.grouper.group(batch)
            # We updated SimpleGrouper to have a group method or we can call forward with n
            # Assuming SimpleGrouper.group alias exists or we use forward
            if hasattr(self.grouper, 'group'):
                grouped_embs = self.grouper.group(batch)
            else:
                grouped_embs = self.grouper(self.n, batch)
            outputs.append(grouped_embs)
        return outputs

    def _score_groups(self, group_embeddings: torch.Tensor) -> torch.Tensor:
        """
        4th Step
        """
        output = []
        for combos in group_embeddings:
            scores = self.scorer.forward(combos)
            output.append(scores)
        return output

    def _batch_embeddings(self, embeddings):
        batched_embeddings = embeddings.clone()
        return batched_embeddings.reshape(self.B, self.n, 384)

    def get_word_embeddings(self, game_id):
        """
        Helper to get embeddings for a single game
        """
        embeddings = self.embedder.encode(self.df[self.df["Game ID"] == game_id]["Word"].tolist())
        return embeddings

    def all_df_to_game_states(self):
        """
        Puts each board from the df into a BoardTensors class,
        which takes in the 16 words as torch embeddings, their
        category tensor, and all the actions you can do with
        this board.
        """
        self.get_all_word_embeddings() # (16 * number of games, 384)

        # Go through each 16 words in a board
        idx = 0
        for game_id, game_df in self.df.groupby("Game ID"):
            game_embeddings = self.embeddings[idx:idx + 16]
            # Since we presorted our df
            categories = torch.arange(4, device=self.device).repeat_interleave(4) # (16,)
            combos = get_all_combos(self.n, 4, self.device)
            board_tensors = BoardTensors(game_embeddings, categories, combos)
            self.board_tensors.append(board_tensors)
            idx += 16
        return self.board_tensors

    def df_to_game(self, game_id):
        """
        Function to put a single board into a BoardTensor.
        """
        game_embeddings = self.get_word_embeddings(game_id)
        categories = torch.arange(4, device=self.device).repeat_interleave(4)
        combos = get_all_combos(self.n, 4, self.device)
        board_tensors = BoardTensors(game_embeddings, categories, combos)
        return board_tensors
