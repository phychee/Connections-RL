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

    def push(self, s, a, r, sp):
        """
        Store the transitions in a queue
        """
        if self.s is None:
            self.s = torch.zeros((self.cap, *s.shape), device=self.device, dtype=s.dtype)
            self.a = torch.zeros((self.cap, *a.shape), device=self.device, dtype=a.dtype)
            self.r = torch.zeros((self.cap, *r.shape), device=self.device, dtype=r.dtype)
            self.sp = torch.zeros((self.cap, *sp.shape), device=self.device, dtype=sp.dtype)

        self.s[self.idx] = s
        self.a[self.idx] = a
        self.r[self.idx] = r
        self.sp[self.idx] = sp

        self.idx = (self.idx + 1) % self.cap
        if not self.idx:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.cap if self.full else self.idx
        sample_idx = torch.randint(0, max_idx, (batch_size,), device=self.device)

        return (
            self.s[sample_idx],
            self.a[sample_idx],
            self.r[sample_idx],
            self.sp[sample_idx]
        )

    def __len__(self):
        return self.cap if self.full else self.idx
