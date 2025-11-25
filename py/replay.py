import torch

class ReplayMemory:
    """
    Class so that the DQN can store the transitions it encountered.
    This code was inspired by the following PyTorch article:
    https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity, device):
        self.cap = capacity
        self.device = device
        self.idx = 0
        self.full = False
        
        # Initialize storage as None
        self.s = None
        self.a = None
        self.r = None
        self.sp = None

    def push(self, state, action, reward, next_state, finished):
        """
        Store the transitions in a queue
        state: dict {'board': (16, D), 'lives': (1,), 'num_groups_found': (1,)}
        action: (1,)
        reward: (1,)
        next_state: dict {'board': (16, D), 'lives': (1,), 'num_groups_found': (1,)}
        finished: (1,) bool
        """
        if self.s is None:
            self.s = torch.zeros((self.cap, *state['board'].shape), device=self.device, dtype=state['board'].dtype)
            self.lives = torch.zeros((self.cap, *state['lives'].shape), device=self.device, dtype=state['lives'].dtype)
            self.num_groups_found = torch.zeros((self.cap, *state['num_groups_found'].shape), device=self.device, dtype=state['num_groups_found'].dtype)
            self.a = torch.zeros((self.cap, *action.shape), device=self.device, dtype=action.dtype)
            self.r = torch.zeros((self.cap, *reward.shape), device=self.device, dtype=reward.dtype)
            self.sp = torch.zeros((self.cap, *next_state['board'].shape), device=self.device, dtype=next_state['board'].dtype)
            self.next_lives = torch.zeros((self.cap, *next_state['lives'].shape), device=self.device, dtype=next_state['lives'].dtype)
            self.next_num_groups_found = torch.zeros((self.cap, *next_state['num_groups_found'].shape), device=self.device, dtype=next_state['num_groups_found'].dtype)
            self.finished = torch.zeros((self.cap, *finished.shape), device=self.device, dtype=finished.dtype)

        self.s[self.idx] = state['board']
        self.lives[self.idx] = state['lives']
        self.num_groups_found[self.idx] = state['num_groups_found']
        self.a[self.idx] = action
        self.r[self.idx] = reward
        self.sp[self.idx] = next_state['board']
        self.next_lives[self.idx] = next_state['lives']
        self.next_num_groups_found[self.idx] = next_state['num_groups_found']
        self.finished[self.idx] = finished

        self.idx = (self.idx + 1) % self.cap
        if not self.idx:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.cap if self.full else self.idx
        # Ensure we have enough samples
        if max_idx < batch_size:
             batch_size = max_idx
             
        sample_idx = torch.randint(0, max_idx, (batch_size,), device=self.device)

        state = {
            'board': self.s[sample_idx],
            'lives': self.lives[sample_idx],
            'num_groups_found': self.num_groups_found[sample_idx]
        }
        
        next_state = {
            'board': self.sp[sample_idx],
            'lives': self.next_lives[sample_idx],
            'num_groups_found': self.next_num_groups_found[sample_idx]
        }

        return (
            state,
            self.a[sample_idx],
            self.r[sample_idx],
            next_state,
            self.finished[sample_idx]
        )

    def __len__(self):
        return self.cap if self.full else self.idx
