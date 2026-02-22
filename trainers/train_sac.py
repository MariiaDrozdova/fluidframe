import os
from typing import Optional, List, Dict

import torch
import numpy as np
from tqdm import tqdm

from agents.agent_saclearning import SACAgent, SACConfig, ReplayBuffer, map_action_to_env

_SAC_SAVE_FOLDER = "./checkpoints/"


def save_sac_checkpoint(agent: "SACAgent", episode: int) -> None:
    os.makedirs(_SAC_SAVE_FOLDER, exist_ok=True)
    path = os.path.join(_SAC_SAVE_FOLDER, f"sac_ep{episode}.pt")
    payload = {
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_targ": agent.q1_targ.state_dict(),
        "q2_targ": agent.q2_targ.state_dict(),
        "cfg": agent.cfg.__dict__,
    }

    torch.save(payload, path)
    print(f"Saved SAC checkpoint: {path}")


def train_sac(
    env,
    n_episodes: int,
    n_steps: int,
    save: bool = False,
    logging: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Train SAC for the continuous swimmer environment.

    We condition the reward by dividing by dt (turns per-step displacement into
        a velocity-like signal; same optimal policy, better scale).
    """

    obs0 = np.asarray(env.reset(), dtype=np.float32)
    obs_dim = int(obs0.shape[0])

    cfg = SACConfig(obs_dim=obs_dim, act_dim=1)

    agent = SACAgent(cfg, seed=seed)

    buf = ReplayBuffer(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.act_dim,
        size=min(500_000, n_episodes * n_steps),
    )

    rng = np.random.default_rng(seed=seed)
    episode_returns: List[float] = []

    total_steps = 0

    for episode in tqdm(range(n_episodes)):
        obs = np.asarray(env.reset(), dtype=np.float32)
        episode_return = 0.0

        for step in range(n_steps):
            # --- action selection ---
            if total_steps < cfg.start_steps:
                a = float(rng.uniform(-1.0, 1.0))
            else:
                a = float(agent.act(obs, deterministic=False))

            env_action = float(map_action_to_env(a))

            # --- step env ---
            next_obs, reward = env.step(env_action)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            # reward
            dt = float(getattr(env, "dt", 1.0))
            reward = float(reward) / dt

            # episode boundary treated as terminal for training
            done = float(step == n_steps - 1)

            # --- store transition ---
            buf.add(obs, np.array([a], dtype=np.float32), reward, next_obs, done)

            # --- advance ---
            obs = next_obs
            episode_return += reward
            total_steps += 1

            # --- updates---
            if total_steps >= cfg.update_after and (total_steps % cfg.update_every == 0):
                for _ in range(cfg.update_every * cfg.updates_per_step):
                    batch = buf.sample(cfg.batch_size)
                    agent.update(batch)

        episode_returns.append(episode_return)

        if save and (episode % 25 == 0 or episode == n_episodes - 1):
            save_sac_checkpoint(agent, episode)
            if episode == n_episodes - 1:
                os.makedirs(_SAC_SAVE_FOLDER, exist_ok=True)
                np.save(
                    os.path.join(_SAC_SAVE_FOLDER, "sac_episode_returns.npy"),
                    np.array(episode_returns, dtype=np.float32),
                )

        if logging:
            if episode % 1 == 0:
                ma = float(np.mean(episode_returns[-20:])) if len(episode_returns) >= 20 else float("nan")
                print(f"Episode {episode} return:\t {episode_return:.4f}\t MA20={ma:.4f}")

    return {"episode_returns": np.array(episode_returns, dtype=np.float32)}