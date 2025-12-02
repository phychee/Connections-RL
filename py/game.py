import torch
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple
from config import GameConfig

@dataclass(frozen=True)
class BoardTensors:
    words: torch.Tensor # (16, D) float
    group_labels: torch.LongTensor # (16,) int in {0, 1, 2, 3}
    combos: torch.LongTensor # (1820, 4) long

@dataclass(frozen=True)
class GameState:
    words_mask: torch.BoolTensor # (16,) - words that we have remaining
    found_groups: torch.BoolTensor # (4,) - groups that we have already found
    lives: torch.Tensor # () int32 - remaining lives
    actions_mask: torch.BoolTensor # (1820,) - actions that we used

    @staticmethod
    def game_start(board: BoardTensors, config: GameConfig) -> "GameState":
        device = board.words.device

        words = torch.ones(16, dtype=torch.bool, device=device)
        groups = torch.zeros(4, dtype=torch.bool, device=device)
        lives = torch.tensor(config.init_lives, dtype=torch.long, device=device)
        actions = torch.ones(board.combos.size(0), dtype=torch.bool, device=device)

        return GameState(
            words_mask = words,
            found_groups = groups,
            lives = lives,
            actions_mask = actions,
        )

class Game:

    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()

    @torch.no_grad()
    def make_guess(self,
                   board: BoardTensors,
                   state: GameState,
                   action_id: int
    ) -> Tuple[GameState, torch.Tensor, bool, Dict]:
        """
        Function that updates the states after the agent
        makes a guess for a potential group.

        Returns: (next_state, reward, is_finished, logging)
        """
        device = board.words.device
        finished = False
        logs = {}

        # Get the group of words that we guessed
        combo = board.combos[action_id]

        # Check that all words picked by action are live
        if not state.words_mask.index_select(0, combo).all():

            # Penalize the illegal action same as a wrong guess, but
            # don't decrease live count
            reward = torch.tensor(self.config.R4_wrong, device=device)

            return (state, reward, False, {"result": "illegal"})

        # Get the category labels for each word
        labels = board.group_labels.index_select(0, combo) # (4,)

        # Get the number of words that are in each label
        group_counts = torch.bincount(labels, minlength=4) # (4,)

        max_count, group = group_counts.max(0)
        group_idx = int(group.item())

        # If we have all 4 words in a label and we haven't discovered the group yet
        if max_count.item() == 4:
            reward = torch.tensor(self.config.R2_correct, device=device)

            # Update the mask for the remaining words
            new_words_mask = state.words_mask.clone()
            new_words_mask[combo] = False

            # Update the mask for the found groups
            new_found_groups = state.found_groups.clone()
            new_found_groups[group_idx] = True

            # If we found 3 groups, the last one is implicitly found
            if new_found_groups.sum() >= 3:
                new_found_groups[:] = True
                new_words_mask[:] = False

            # Update the action mask to remove actions that use words that we already used
            # and the action we took at this step
            new_actions_mask = self._remove_used_action(board, state, action_id)
            no_used_words_mask = self._remove_actions_with_used_words(board, state, new_words_mask)

            new_actions_mask &= no_used_words_mask

            # If we found all groups this means we won
            if new_found_groups.all():
                reward = torch.tensor(self.config.R1_win, device=device)
                finished = True

            next_state = replace(
                state,
                words_mask = new_words_mask,
                found_groups = new_found_groups,
                actions_mask = new_actions_mask,
            )
            return (next_state, reward, finished, {"result": "correct", "category": group_idx})

        # If we are one-away
        elif max_count.item() == 3:
            reward = torch.tensor(self.config.R3_one_away, device=device)
            new_lives = state.lives - 1
            game_over = new_lives.item() <= 0
            new_actions_mask = self._remove_used_action(board, state, action_id)

            next_state = replace(
                state,
                lives=new_lives,
                actions_mask=new_actions_mask,
            )

            if game_over:
                reward = torch.tensor(self.config.R5_game_over, device=device)
                finished = True

            return (next_state, reward, finished, {"result": "one_away", "category": group_idx})

        # If we made the wrong guess
        else:
            reward = torch.tensor(self.config.R4_wrong, device=device)
            new_lives = state.lives - 1
            game_over = new_lives.item() <= 0
            new_actions_mask = self._remove_used_action(board, state, action_id)

            next_state = replace(
                state,
                lives = new_lives,
                actions_mask=new_actions_mask,
            )

            if game_over:
                reward = torch.tensor(self.config.R5_game_over, device=device)
                finished = True

            return (next_state, reward, finished, {"result": "wrong"})

    def _remove_used_action(self,
                            board: BoardTensors,
                            state: GameState,
                            picked_action: int
                           ) -> torch.BoolTensor:
        new_actions_mask = state.actions_mask.clone()
        new_actions_mask[picked_action] = False
        return new_actions_mask

    def _remove_actions_with_used_words(self,
                            board: BoardTensors,
                            state: GameState,
                            new_words_mask: torch.BoolTensor
                           ) -> torch.BoolTensor:
        # Get all the possible word combos
        combos = board.combos

        # Create a mask that zeros out combos that have words that were already used
        no_used_words = new_words_mask[combos].all(dim=1)

        return no_used_words
