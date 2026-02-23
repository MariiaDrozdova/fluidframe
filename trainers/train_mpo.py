import os
from typing import Optional, List, Dict

import torch
import numpy as np
from tqdm import tqdm

from agents.agent_mpolearning import MPOAgent, MPOConfig
from agents.agent_saclearning import map_action_to_env, ReplayBuffer

_MPO_SAVE_FOLDER = "./checkpoints/"


def save_mpo_checkpoint(agent: "MPOAgent", episode: int) -> None:
    os.makedirs(_MPO_SAVE_FOLDER, exist_ok=True)
    path = os.path.join(_MPO_SAVE_FOLDER, f"mpo_ep{episode}.pt")
    payload = {
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_targ": agent.q1_targ.state_dict(),
        "q2_targ": agent.q2_targ.state_dict(),
        "cfg": agent.cfg.__dict__,
    }
    torch.save(payload, path)
    print(f"Saved MPO checkpoint: {path}")

def save_mpo_checkpoint(
    agent: "MPOAgent",
    episode: int,
    swimmer_speed: float,
    alignment_timescale: float,
    seed: int,
    observation_type: str,
    tag: str = "",
) -> None:
    os.makedirs(_MPO_SAVE_FOLDER, exist_ok=True)

    name = (
        f"mpo_{tag}_ep{episode}"
        f"_phi{swimmer_speed}"
        f"_psi{alignment_timescale}"
        f"_seed{seed}"
        f"_obs{observation_type}.pt"
    )

    path = os.path.join(_MPO_SAVE_FOLDER, name)

    payload = {
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_targ": agent.q1_targ.state_dict(),
        "q2_targ": agent.q2_targ.state_dict(),
        "cfg": agent.cfg.__dict__,
        "episode": episode,
        "swimmer_speed": swimmer_speed,
        "alignment_timescale": alignment_timescale,
        "seed": seed,
        "observation_type": observation_type,
    }

    torch.save(payload, path)
    print(f"Saved MPO checkpoint: {path}")


def train_mpo(
    env,
    n_episodes: int,
    n_steps: int,
    save: bool = False,
    logging: bool = True,
    seed: Optional[int] = None,
    use_wandb: bool = True,
) -> Dict[str, np.ndarray]:
    """Train MPO for the continuous swimmer environment (without Retrace).

    - Off-policy replay buffer
    - TD(0) critic targets with EMA target networks
    - MPO E/M-step policy improvement inside agent.update()
    - Reward normalization by dt for scale consistency (same optimal policy, better scale)
    """

    obs0 = np.asarray(env.reset(), dtype=np.float32)
    obs_dim = int(obs0.shape[0])

    cfg = MPOConfig(obs_dim=obs_dim, act_dim=1)
    agent = MPOAgent(cfg, seed=seed)

    if use_wandb:
        import wandb
        wandb.init(
            project="fluidframe",
            name=f"mpo_phi{getattr(env, 'swimmer_speed', '?')}_psi{getattr(env, 'alignment_timescale', '?')}",
            config=cfg.__dict__,
        )
        wandb.watch(agent.actor, log="gradients", log_freq=2000)
        wandb.watch(agent.q1, log="gradients", log_freq=2000)

    buf = ReplayBuffer(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.act_dim,
        size=min(500_000, n_episodes * n_steps),
    )

    rng = np.random.default_rng(seed=seed)
    episode_returns: List[float] = []

    total_steps = 0
    best_ma = float("-inf")

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

            # Optional: normalize by dt for scale consistency
            dt = float(getattr(env, "dt", 1.0))
            c = 1.0 / dt  # reward scaling factor
            reward = float(reward) *c

            # episode boundary treated as terminal for training
            done = float(step == n_steps - 1)

            # --- store transition ---
            buf.add(obs, np.array([a], dtype=np.float32), reward, next_obs, done)

            # --- advance ---
            obs = next_obs
            episode_return += reward
            total_steps += 1

            # --- updates ---
            if total_steps >= cfg.update_after and (total_steps % cfg.update_every == 0):
                for _ in range(cfg.update_every * cfg.updates_per_step):
                    batch = buf.sample(cfg.batch_size)
                    info = agent.update(batch)
                    if use_wandb:
                        wandb.log(info, step=total_steps)

        episode_returns.append(episode_return)
        ma = float(np.mean(episode_returns[-20:]))/c if len(episode_returns) >= 20 else float("nan")

        if ma > best_ma:
            best_ma = ma
            save_mpo_checkpoint(
                agent,
                episode=episode,
                swimmer_speed=env.swimmer_speed,
                alignment_timescale=env.alignment_timescale,
                seed=env.seed,
                observation_type=getattr(env, "observation_type", "unknown"),
                tag="BEST",
            )


        if logging:
            print(f"Episode {episode} return:\t {episode_return:.4f}\t MA20={ma:.4f}")
            if use_wandb:
                wandb.log(
                    {
                        "episode_return": episode_return,
                        "ma20_return": ma,
                        "episode": episode,
                    },
                    step=total_steps,
                )

    return {"episode_returns": np.array(episode_returns, dtype=np.float32)}