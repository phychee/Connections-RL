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

    def push(self, s, lives, num_groups_found, a, r, sp, next_lives, next_num_groups_found, finished):
        """
        Store the transitions in a queue
        s: (16, D)
        lives: (1,) float
        num_groups_found: (1,) float
        a: (1,)
        r: (1,)
        sp: (16, D)
        next_lives: (1,) float
        next_num_groups_found: (1,) float
        finished: (1,) bool
        """
        if self.s is None:
            self.s = torch.zeros((self.cap, *s.shape), device=self.device, dtype=s.dtype)
            self.lives = torch.zeros((self.cap, *lives.shape), device=self.device, dtype=lives.dtype)
            self.num_groups_found = torch.zeros((self.cap, *num_groups_found.shape), device=self.device, dtype=num_groups_found.dtype)
            self.a = torch.zeros((self.cap, *a.shape), device=self.device, dtype=a.dtype)
            self.r = torch.zeros((self.cap, *r.shape), device=self.device, dtype=r.dtype)
            self.sp = torch.zeros((self.cap, *sp.shape), device=self.device, dtype=sp.dtype)
            self.next_lives = torch.zeros((self.cap, *next_lives.shape), device=self.device, dtype=next_lives.dtype)
            self.next_num_groups_found = torch.zeros((self.cap, *next_num_groups_found.shape), device=self.device, dtype=next_num_groups_found.dtype)
            self.finished = torch.zeros((self.cap, *finished.shape), device=self.device, dtype=finished.dtype)

        self.s[self.idx] = s
        self.lives[self.idx] = lives
        self.num_groups_found[self.idx] = num_groups_found
        self.a[self.idx] = a
        self.r[self.idx] = r
        self.sp[self.idx] = sp
        self.next_lives[self.idx] = next_lives
        self.next_num_groups_found[self.idx] = next_num_groups_found
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

        return (
            self.s[sample_idx],
            self.lives[sample_idx],
            self.num_groups_found[sample_idx],
            self.a[sample_idx],
            self.r[sample_idx],
            self.sp[sample_idx],
            self.next_lives[sample_idx],
            self.next_num_groups_found[sample_idx],
            self.finished[sample_idx]
        )

    def __len__(self):
        return self.cap if self.full else self.idx
