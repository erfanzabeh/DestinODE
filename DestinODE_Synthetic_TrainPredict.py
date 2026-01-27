
# =============================
# DestinODE: Full Synthetic Experiment (Train on 0:n, Predict n+1)
# =============================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================
# Step 1: Generate Synthetic Data
# =============================

def generate_synthetic_dataset(T=10, S=5, K=40, N=50, r=3):
    W0 = 0.1 * torch.randn(N, N)
    U, _, Vt = torch.linalg.svd(W0)
    U = U[:, :r]
    V = Vt[:r, :].T

    B = 0.05 * torch.randn(N, 1)
    b = 0.1 * torch.randn(N)
    R = torch.randn(1, N) * 0.05

    t_vals = torch.linspace(0, 2 * np.pi, T)
    z = torch.stack([torch.sin(t_vals + np.pi * i / r) + 0.1 * t_vals for i in range(r)], dim=1)

    x_full = torch.zeros(T, S, K, N)
    v_full = torch.zeros(T, S, K)
    u_full = torch.zeros(T, S, K)

    for t in range(T):
        Wt = W0 + U @ torch.diag(z[t]) @ V.T
        for s in range(S):
            v = torch.randn(K) * 0.5
            x = torch.zeros(K, N)
            x[0] = torch.randn(N) * 0.1
            for k in range(K - 1):
                x[k + 1] = torch.tanh(Wt @ x[k] + (B * v[k]).squeeze() + b)
            u_hat = (R @ x.T).squeeze()
            x_full[t, s] = x
            v_full[t, s] = v
            u_full[t, s] = u_hat + 0.05 * torch.randn(K)
    return x_full, v_full, u_full, W0, U, V, z, B, b, R

# =============================
# Step 2: Dataset class
# =============================

class SplitSessionDataset(Dataset):
    def __init__(self, x, v, u, train_days):
        self.x = x
        self.v = v
        self.u = u
        self.train_days = train_days
        self.S = x.shape[1]

    def __len__(self):
        return len(self.train_days) * self.S

    def __getitem__(self, idx):
        t_idx = self.train_days[idx // self.S]
        s = idx % self.S
        return self.x[t_idx, s], self.v[t_idx, s], self.u[t_idx, s], t_idx

# =============================
# Step 3: Model
# =============================

class SlowODE(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(r + 1, 64),
            nn.Tanh(),
            nn.Linear(64, r)
        )

    def forward(self, t, z):
        if isinstance(t, torch.Tensor):
            t = t.item()
        tvec = torch.full_like(z[:, :1], t)
        return self.net(torch.cat([z, tvec], dim=1))

class PINN_RNN(nn.Module):
    def __init__(self, N, r, W0, learn_UV=False):
        super().__init__()
        self.W0 = nn.Parameter(W0)
        U, _, Vt = torch.linalg.svd(W0)
        self.U = nn.Parameter(U[:, :r]) if learn_UV else U[:, :r].detach()
        self.V = nn.Parameter(Vt[:r, :].T) if learn_UV else Vt[:r, :].T.detach()

        self.z0 = nn.Parameter(torch.zeros(r))
        self.slow = SlowODE(r)
        self.B = nn.Parameter(torch.randn(N, 1) * 0.01)
        self.R = nn.Parameter(torch.randn(1, N) * 0.01)
        self.b_fixed = None

    def set_fixed_bias(self, b):
        self.b_fixed = b

    def rnn_cell(self, x, v, W):
        return torch.tanh((W @ x) + (self.B * v).squeeze() + self.b_fixed)

    def weights_over_time(self, T):
        times = torch.arange(T, dtype=torch.float32)
        z0_cpu = self.z0.unsqueeze(0).float()
        z_cpu = odeint(self.slow, z0_cpu, times).squeeze(1)
        return self.W0 + self.U @ torch.diag_embed(z_cpu) @ self.V.T

    def forward(self, x0, v, t, Ws):
        W = Ws[t]
        K, N = x0.shape
        x_list = [x0[0]]
        rec = dec = upd = 0
        for s in range(K - 1):
            up = self.rnn_cell(x_list[s], v[s], W)
            rec += (x_list[s] - x0[s]).pow(2).sum()
            dec += (self.R @ x_list[s] - v[s].unsqueeze(0)).pow(2).sum()
            upd += (up - self.rnn_cell(x_list[s], v[s], W)).pow(2).sum()
            x_list.append(up)
        return rec, dec, upd

# =============================
# Step 4: Run experiment
# =============================

x, v, u, W0, U, V, z_true, B, b, R = generate_synthetic_dataset()
T, S, K, N = x.shape
train_days = list(range(7))

dataset = SplitSessionDataset(x, v, u, train_days)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = PINN_RNN(N=N, r=3, W0=W0, learn_UV=True)
model.set_fixed_bias(b)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(20):
    Ws = model.weights_over_time(T)
    L = 0
    for x_batch, v_batch, u_batch, t_batch in loader:
        loss = 0
        for i in range(x_batch.shape[0]):
            Rloss, Dloss, Uloss = model(x_batch[i], v_batch[i], t_batch[i], Ws)
            loss += Rloss + Dloss + 0.01 * Uloss
        loss /= x_batch.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        L += loss.item()
    print(f"Epoch {ep:02d} | Loss: {L/len(loader):.4f}")
