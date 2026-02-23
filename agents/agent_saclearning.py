import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional

# Map SAC action in [-1,1] -> environment action in [0,2*pi)
def map_action_to_env(a_tanh: float) -> float:
    a_tanh = float(np.clip(a_tanh, -1.0, 1.0))
    return (a_tanh + 1.0) * np.pi % (2*np.pi)

# ---------- replay buffer ----------
class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int = 200_000):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.size, self.ptr, self.max_size = 0, 0, size

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.next_obs[self.ptr] = o2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = 256):
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=torch.tensor(self.obs[idx]),
            acts=torch.tensor(self.acts[idx]),
            rews=torch.tensor(self.rews[idx]),
            next_obs=torch.tensor(self.next_obs[idx]),
            done=torch.tensor(self.done[idx]),
        )

# ---------- networks ----------
LOG_STD_MIN, LOG_STD_MAX = -10, 2

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)

        # log_prob with tanh correction
        log_prob = (-0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2*log_std + np.log(2*np.pi))).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, log_prob

class CriticQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))

# ---------- SAC agent ----------
@dataclass
class SACConfig:
    obs_dim: int = 9
    act_dim: int = 1
    gamma: float = 0.997
    tau: float = 0.005
    alpha: float = 0.01
    lr: float = 1e-4
    batch_size: int = 256
    start_steps: int =10_000
    update_after: int = 10_000
    update_every: int = 50
    updates_per_step: int = 1

class SACAgent:
    def __init__(self, cfg: SACConfig, seed: Optional[int] = None):
        self.cfg = cfg
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.actor = Actor(cfg.obs_dim, cfg.act_dim)
        self.q1 = CriticQ(cfg.obs_dim, cfg.act_dim)
        self.q2 = CriticQ(cfg.obs_dim, cfg.act_dim)
        self.q1_targ = CriticQ(cfg.obs_dim, cfg.act_dim)
        self.q2_targ = CriticQ(cfg.obs_dim, cfg.act_dim)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> float:
        obs = np.asarray(obs, dtype=np.float32)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mu, _ = self.actor(obs_t)
        if deterministic:
            a = torch.tanh(mu)
        else:
            a, _ = self.actor.sample(obs_t)
        return float(a.squeeze().cpu().numpy())

    def update(self, batch):
        o, a, r, o2, d = batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["done"]

        # --- critic target ---
        with torch.no_grad():
            a2, logp_a2 = self.actor.sample(o2)
            q1_t = self.q1_targ(o2, a2)
            q2_t = self.q2_targ(o2, a2)
            q_t = torch.min(q1_t, q2_t)
            backup = r + self.cfg.gamma * (1 - d) * (q_t - self.cfg.alpha * logp_a2)

        # --- critic loss ---
        q1_val = self.q1(o, a)
        q2_val = self.q2(o, a)
        q_loss = F.mse_loss(q1_val, backup) + F.mse_loss(q2_val, backup)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # --- actor loss ---
        a_pi, logp_a = self.actor.sample(o)
        q1_pi = self.q1(o, a_pi)
        q2_pi = self.q2(o, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = (self.cfg.alpha * logp_a - q_pi).mean()

        self.actor_opt.zero_grad()
        pi_loss.backward()
        self.actor_opt.step()

        # --- target update ---
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        # ----- diagnostics -----
        with torch.no_grad():
            # entropy proxy = -log pi
            entropy = (-logp_a).mean()

            # action stats (tanh space)
            a_mean = a_pi.mean()
            a_std = a_pi.std()
            a_abs = a_pi.abs().mean()

            # Q stats
            q1m, q2m = q1_val.mean(), q2_val.mean()
            qmin = torch.min(q1_val, q2_val).mean()
            targ_m = backup.mean()
            td_abs = (backup - qmin.unsqueeze(-1)).abs().mean()  # rough proxy

        return {
            "q_loss": float(q_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "entropy": float(entropy.item()),
            "a_mean": float(a_mean.item()),
            "a_std": float(a_std.item()),
            "a_abs": float(a_abs.item()),
            "q1_mean": float(q1m.item()),
            "q2_mean": float(q2m.item()),
            "qmin_mean": float(qmin.item()),
            "target_mean": float(targ_m.item()),
            "td_abs": float(td_abs.item()),
        }