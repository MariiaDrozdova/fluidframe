import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional

from agents.agent_saclearning import Actor, CriticQ, map_action_to_env

# ---------- networks ----------
LOG_STD_MIN, LOG_STD_MAX = -10, 2

@dataclass
class MPOConfig:
    obs_dim: int = 9
    act_dim: int = 1

    # critic / target
    gamma: float = 0.99
    tau: float = 0.005
    lr_q: float = 1e-4

    # policy / MPO
    lr_pi: float = 3e-4
    n_action_samples: int = 10          # samples per state for E-step
    eps_eta: float = 0.01               # KL(q || pi_old) constraint (E-step)
    eps_mu: float = 1e-3                # mean KL constraint proxy (M-step)
    eps_sigma: float = 1e-3             # std KL constraint proxy (M-step)
    eta_min: float = 1e-3
    eta_max: float = 1e3
    eta_iters: int = 10                 # bisection iterations
    kl_coef: float = 1.0                # soft penalty strength for M-step KL
    m_steps: int = 5                  # number of policy gradient steps per update (M-step)

    # training loop expectations
    batch_size: int = 256
    start_steps: int = 10_000
    update_after: int = 10_000
    update_every: int = 50
    updates_per_step: int = 1


class MPOAgent:
    """A practical MPO-style agent:

    - Off-policy twin Q critics with Polyak target networks (TD(0) backup).
    - MPO-style policy improvement:
        * E-step: sample multiple actions per state, compute Q, compute weights via softmax(Q/eta).
          eta is solved (approximately) via bisection on the dual to satisfy a KL(q||pi_old) budget.
        * M-step: weighted maximum-likelihood update of the policy, with a soft KL penalty to the
          previous policy to avoid collapsing variance / overly large steps.

    Notes:
    - This is a simplified MPO: we do not implement Retrace; TD(0) is typically stable
      enough for small environments.
    - We keep the same squashed Gaussian actor used for SAC (tanh + log-prob correction).
    """

    def __init__(self, cfg: MPOConfig, seed: Optional[int] = None):
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

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_pi)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr_q
        )

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

    # ---------- internal: dual for eta ----------
    def _solve_eta(self, q_values: torch.Tensor) -> float:
        """Solve eta (temperature) for weights softmax(Q/eta) using a bisection on the convex dual.

        q_values: Tensor of shape [B, N] (B states, N sampled actions)

        We use a standard REPS/MPO-style dual:
            g(eta) = eta * eps + eta * mean_s log( mean_a exp((Q - maxQ)/eta) ) + mean_s maxQ

        This keeps the exp numerically stable.
        """
        eps = float(self.cfg.eps_eta)

        # detach: eta solving should not backprop
        Q = q_values.detach()
        Q_max, _ = Q.max(dim=1, keepdim=True)               # [B, 1]
        Qc = Q - Q_max                                      # [B, N]

        eta_lo = float(self.cfg.eta_min)
        eta_hi = float(self.cfg.eta_max)

        # helper: compute dual value
        def dual(eta: float) -> float:
            e = torch.exp(Qc / eta).mean(dim=1)             # [B]
            val = eta * eps + eta * torch.log(e + 1e-12)    # [B]
            val = val + Q_max.squeeze(1)                    # [B]
            return float(val.mean().cpu().item())

        # bisection on derivative sign (approx) using finite differences
        # (cheap and robust for 1D scalar)
        for _ in range(int(self.cfg.eta_iters)):
            mid = 0.5 * (eta_lo + eta_hi)
            d = 1e-4 * max(1.0, mid)
            g1 = dual(max(self.cfg.eta_min, mid - d))
            g2 = dual(min(self.cfg.eta_max, mid + d))
            grad = (g2 - g1) / (2.0 * d)
            # For a convex dual, grad>0 means we're to the right of optimum
            if grad > 0:
                eta_hi = mid
            else:
                eta_lo = mid
        return float(0.5 * (eta_lo + eta_hi))

    # ---------- internal: analytic KL between two diagonal Gaussians in pre-tanh space ----------
    @staticmethod
    def _gaussian_kl(mu0: torch.Tensor, logstd0: torch.Tensor, mu1: torch.Tensor, logstd1: torch.Tensor) -> torch.Tensor:
        """KL( N(mu0,std0) || N(mu1,std1) ) for diagonal Gaussians. Returns [B, 1]."""
        std0 = torch.exp(logstd0)
        std1 = torch.exp(logstd1)
        var0 = std0.pow(2)
        var1 = std1.pow(2)
        kl = (logstd1 - logstd0) + (var0 + (mu0 - mu1).pow(2)) / (2.0 * var1 + 1e-12) - 0.5
        return kl.sum(dim=-1, keepdim=True)

    def update(self, batch):
        """One combined update: critics (TD(0)) + MPO policy improvement (E/M-step)."""
        o, a, r, o2, d = batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["done"]

        # =====================
        # Critic update (TD(0))
        # =====================
        with torch.no_grad():
            # Use current policy to sample next action; no entropy term here (MPO isn't max-entropy).
            a2, _ = self.actor.sample(o2)
            q1_t = self.q1_targ(o2, a2)
            q2_t = self.q2_targ(o2, a2)
            q_t = torch.min(q1_t, q2_t)
            backup = r + self.cfg.gamma * (1 - d) * q_t

        q1_val = self.q1(o, a)
        q2_val = self.q2(o, a)
        q_loss = F.smooth_l1_loss(q1_val, backup) + F.smooth_l1_loss(q2_val, backup)

        self.q_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.q_opt.step()

        # =====================
        # MPO policy improvement
        # =====================
        # Snapshot old policy (for KL penalty)
        with torch.no_grad():
            mu_old, logstd_old = self.actor(o)

        # ----- E-step: sample multiple actions per state and compute weights -----
        B = o.shape[0]
        N = int(self.cfg.n_action_samples)

        # sample N actions per state using current (old) policy (NO grad)
        o_rep = o.unsqueeze(1).expand(B, N, o.shape[-1]).reshape(B * N, o.shape[-1])
        with torch.no_grad():
            a_samp, _ = self.actor.sample(o_rep)  # [B*N, act]

        # compute Q for each sampled action (NO grad)
        with torch.no_grad():
            q1_s = self.q1(o_rep, a_samp)
            q2_s = self.q2(o_rep, a_samp)
            q_s = torch.min(q1_s, q2_s).reshape(B, N)  # [B, N]

        # solve eta and compute normalized weights
        eta = self._solve_eta(q_s)
        w = torch.softmax(
            (q_s - q_s.max(dim=1, keepdim=True).values) / max(eta, 1e-8),
            dim=1,
        ).detach()  # [B, N]

        # ----- M-step: weighted maximum likelihood with KL penalty -----
        # We take multiple policy gradient steps so the KL penalty actually constrains the update.
        w_flat = w.reshape(B * N, 1)  # already detached above
        o_flat = o_rep
        a_flat = a_samp.detach()

        # atanh for tanh-squashed actions (clip to avoid inf)
        a_clip = torch.clamp(a_flat, -0.999999, 0.999999).detach()
        pre_tanh = (0.5 * (torch.log1p(a_clip) - torch.log1p(-a_clip))).detach()

        last_pi_loss = None
        kl_old_new = torch.tensor(0.0, device=o.device)

        for _ in range(int(getattr(self.cfg, "m_steps", 1))):
            # log-prob of the sampled actions under *current* policy
            mu, logstd = self.actor(o_flat)
            std = torch.exp(logstd)
            logp = (
                -0.5
                * (
                    ((pre_tanh - mu) / (std + 1e-8)) ** 2
                    + 2 * logstd
                    + np.log(2 * np.pi)
                )
            ).sum(-1, keepdim=True)
            # tanh correction
            logp = logp - torch.log(1 - a_clip.pow(2) + 1e-6).sum(-1, keepdim=True)

            # Weighted negative log-likelihood
            nll = -(w_flat * logp).mean()

            # KL penalty to the fixed old policy (in pre-tanh Gaussian space)
            mu_new_s, logstd_new_s = self.actor(o)
            kl_old_new = self._gaussian_kl(mu_old, logstd_old, mu_new_s, logstd_new_s).mean()

            pi_loss = nll + self.cfg.kl_coef * kl_old_new

            self.actor_opt.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            last_pi_loss = pi_loss

        # for logging
        if last_pi_loss is None:
            last_pi_loss = torch.tensor(0.0, device=o.device)

        # ============
        # Target update
        # ============
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        # ----- diagnostics -----
        with torch.no_grad():
            a_abs = a_samp.abs().mean()
            q_mean = q_s.mean()
            q_std = q_s.std()
            w_max = w.max(dim=1).values.mean()
            td_abs = (backup - torch.min(q1_val, q2_val)).abs().mean()

        return {
            "q_loss": float(q_loss.item()),
            "pi_loss": float(last_pi_loss.item()),
            "eta": float(eta),
            "kl_old_new": float(kl_old_new.item()),
            "w_max": float(w_max.item()),
            "a_abs": float(a_abs.item()),
            "q_mean": float(q_mean.item()),
            "q_std": float(q_std.item()),
            "td_abs": float(td_abs.item()),
        }