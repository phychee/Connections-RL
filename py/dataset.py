import torch
import pandas as pd
from torch.utils.data import Dataset

class ConnectionsDataset(Dataset):
    def __getitem__(self, idx):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError

class ConnectionsData(ConnectionsDataset):
    def __init__(self, data):
        """
        Args:
            data: data, either path or df
        """
        super().__init__()
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise ValueError("Input to ConnectionsData should be path or dataframe")
        self.grouped_data = self.df.groupby("Game ID")
        self.game_ids = list(self.grouped_data.groups.keys())

    def __getitem__(self, idx):
        game_id = self.game_ids[idx]
        game_df = self.grouped_data.get_group(game_id)
        words = game_df["Word"].tolist()
        targets = torch.tensor(game_df["Group Level"].values, dtype=torch.long)
        return {
            "game_id": game_id,
            "words": words,
            "targets": targets,
            "group_names": game_df["Group Name"].tolist()
        }

    def __len__(self):
        return len(self.game_ids)
