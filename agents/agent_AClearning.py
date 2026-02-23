# agent_actorcriticlearning.py
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from agents.agent_saclearning import Actor

LOG_STD_MIN, LOG_STD_MAX = -2, 1


class ValueNet(nn.Module):
    """Simple MLP value function V(s)."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # (B,1)


@dataclass
class ActorCriticConfig:
    obs_dim: int
    act_dim: int = 1
    gamma: float = 0.99
    lr_actor: float = 5e-5
    lr_critic: float = 2e-4
    value_coef: float = 1.0
    entropy_coef: float = 1e-3
    grad_clip: float = 1.0


class ActorCriticAgent:
    """
    On-policy Actor-Critic.
    """

    def __init__(self, cfg: ActorCriticConfig, seed: Optional[int] = None):
        self.cfg = cfg
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.actor = Actor(cfg.obs_dim, cfg.act_dim)
        self.critic = ValueNet(cfg.obs_dim)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        obs = np.asarray(obs, dtype=np.float32)
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1,obs_dim)

    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Returns:
          a_tanh (float): action in [-1,1] as python float (detached)
          logp   (tensor scalar): log pi(a|s) (keeps grad if deterministic=False)
          entropy_proxy (tensor scalar): approx entropy bonus term
        """
        obs_t = self._to_tensor(obs)

        if deterministic:
            with torch.no_grad():
                mu, _ = self.actor(obs_t)
                a = torch.tanh(mu)
            logp = torch.zeros((), dtype=torch.float32)
            entropy_proxy = torch.zeros((), dtype=torch.float32)
        else:
            # NOTE: We only need a sampled action for data collection.
            # `update_batch()` recomputes log-prob for the stored action, so
            # we avoid computing log-prob here (it can become huge and is slow).
            with torch.no_grad():
                mu, log_std = self.actor(obs_t)
                log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
                mu = torch.clamp(mu, -5.0, 5.0)
                std = torch.exp(log_std)
                eps = torch.randn_like(mu)
                pre_tanh = mu + std * eps
                a = torch.tanh(pre_tanh)
            logp = torch.zeros((), dtype=torch.float32)
            entropy_proxy = torch.zeros((), dtype=torch.float32)

        a_float = float(a.squeeze().detach().cpu().numpy())
        return a_float, logp, entropy_proxy

    def _log_prob_tanh_gaussian(
        self, mu: torch.Tensor, log_std: torch.Tensor, a_tanh: torch.Tensor
    ) -> torch.Tensor:
        """Log-prob of tanh-squashed Gaussian policy for given tanh-action.

        Args:
            mu, log_std: tensors of shape (B, act_dim)
            a_tanh: tanh-squashed action in [-1,1], shape (B, act_dim)

        Returns:
            log_prob: shape (B, 1)
        """
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Stable atanh
        a_clamped = torch.clamp(a_tanh, -0.999, 0.999)
        pre_tanh = 0.5 * (torch.log1p(a_clamped) - torch.log1p(-a_clamped))

        # log N(pre_tanh; mu, std)
        log_prob = (
            -0.5
            * (
                ((pre_tanh - mu) / (std + 1e-8)) ** 2
                + 2.0 * log_std
                + np.log(2.0 * np.pi)
            )
        ).sum(-1, keepdim=True)

        # tanh correction
        log_prob = log_prob - torch.log(1.0 - a_clamped.pow(2) + 1e-6).sum(-1, keepdim=True)
        return log_prob

    def update_step(
        self,
        obs: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
        logp: torch.Tensor,
        entropy_proxy: torch.Tensor,
    ) -> Dict[str, float]:
        """One TD(0) actor-critic update."""
        obs_t = self._to_tensor(obs)
        next_obs_t = self._to_tensor(next_obs)

        r = torch.tensor([[float(reward)]], dtype=torch.float32)  # (1,1)
        d = torch.tensor([[float(done)]], dtype=torch.float32)    # (1,1)

        v = self.critic(obs_t)  # (1,1)

        with torch.no_grad():
            v_next = self.critic(next_obs_t)
            target = r + self.cfg.gamma * (1.0 - d) * v_next

        advantage = target - v  # (1,1)


        # Critic update
        critic_loss = 0.5 * (v - target).pow(2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        (self.cfg.value_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()

        # Actor update (recompute log-prob for the stored action `a_tanh` is not available here,
        # so `update_step` is not used for training in this implementation.)
        actor_loss = torch.zeros((), dtype=torch.float32)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "v": float(v.item()),
            "target": float(target.item()),
        }

    def update_batch(self, transitions) -> Dict[str, float]:
        """Perform ONE actor update and ONE critic update from a small on-policy batch.

        transitions: list of tuples (obs, reward, next_obs, done, a_tanh_float)

        We store the sampled action as a float (tanh space) and recompute log-prob under
        the *current* policy to avoid autograd graph / in-place parameter update issues.
        """
        if len(transitions) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        obs_np = np.asarray([t[0] for t in transitions], dtype=np.float32)
        next_obs_np = np.asarray([t[2] for t in transitions], dtype=np.float32)
        rews_np = np.asarray([t[1] for t in transitions], dtype=np.float32).reshape(-1, 1)
        done_np = np.asarray([t[3] for t in transitions], dtype=np.float32).reshape(-1, 1)
        a_np = np.asarray([t[4] for t in transitions], dtype=np.float32).reshape(-1, 1)

        obs = torch.tensor(obs_np, dtype=torch.float32)
        next_obs = torch.tensor(next_obs_np, dtype=torch.float32)
        rews = torch.tensor(rews_np, dtype=torch.float32)
        done = torch.tensor(done_np, dtype=torch.float32)
        a = torch.tensor(a_np, dtype=torch.float32)

        # ----- critic target -----
        v = self.critic(obs)
        with torch.no_grad():
            v_next = self.critic(next_obs)
            target = rews + self.cfg.gamma * (1.0 - done) * v_next

        advantage = target - v

        # ----- critic update -----
        critic_loss = 0.5 * (v - target).pow(2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        (self.cfg.value_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()

        # ----- actor update (recompute log-prob for stored tanh action) -----
        mu, log_std = self.actor(obs)
        log_prob = self._log_prob_tanh_gaussian(mu, log_std, a)

        adv = advantage.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.clamp(-5, 5)

        # Stable entropy bonus (Gaussian entropy; ignores tanh correction but prevents std collapse)
        log_std_c = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        entropy_proxy = (0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std_c).sum(-1, keepdim=True).mean()

        actor_loss = -(log_prob * adv).mean()
        if self.cfg.entropy_coef != 0.0:
            actor_loss = actor_loss - self.cfg.entropy_coef * entropy_proxy

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)

        if not torch.isfinite(actor_loss):
            return {
                "actor_loss": float("nan"),
                "critic_loss": float(critic_loss.item()),
                "v": float(v.mean().item()),
                "target": float(target.mean().item()),
            }

        self.actor_opt.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "v": float(v.mean().item()),
            "target": float(target.mean().item()),
        }