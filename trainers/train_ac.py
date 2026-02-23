# train_actorcritic.py
import os
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm
import torch

from agents.agent_AClearning import ActorCriticAgent, ActorCriticConfig
from agents.agent_saclearning import map_action_to_env

_SAVE_FOLDER = "./checkpoints/"

def save_actorcritic_checkpoint(
    agent: "ActorCriticAgent",
    episode: int,
    swimmer_speed: float,
    alignment_timescale: float,
    seed: int,
    observation_type: str,
    tag: str = "",
) -> None:
    os.makedirs(_SAVE_FOLDER, exist_ok=True)

    name = (
        f"ac_{tag}_ep{episode}"
        f"_phi{swimmer_speed}"
        f"_psi{alignment_timescale}"
        f"_seed{seed}"
        f"_obs{observation_type}.pt"
    )

    path = os.path.join(_SAVE_FOLDER, name)
    payload = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "cfg": agent.cfg.__dict__,
        "episode": episode,
        "swimmer_speed": swimmer_speed,
        "alignment_timescale": alignment_timescale,
        "seed": seed,
        "observation_type": observation_type,
    }
    torch.save(payload, path)
    print(f"Saved ActorCritic checkpoint: {path}")


def train_actorcritic(
    env,
    n_episodes: int,
    n_steps: int,
    save: bool = False,
    logging: bool = True,
    seed: Optional[int] = None,
    use_wandb: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Train simple on-policy Actor-Critic.
    """

    obs0 = np.asarray(env.reset(), dtype=np.float32)
    cfg = ActorCriticConfig(obs_dim=int(obs0.shape[0]), act_dim=1)
    agent = ActorCriticAgent(cfg, seed=seed)

    if use_wandb:
        import wandb
        wandb.init(
            project="fluidframe",
            name=f"ac_phi{getattr(env, 'swimmer_speed', '?')}_psi{getattr(env, 'alignment_timescale', '?')}",
            config=cfg.__dict__,
        )
        wandb.watch(agent.actor, log="gradients", log_freq=2000)
        wandb.watch(agent.critic, log="gradients", log_freq=2000)

    # how many environment steps to collect before updating (to reduce variance)
    update_every = 50
    total_steps = 0

    episode_returns: List[float] = []
    best_ma = float("-inf")

    for episode in tqdm(range(n_episodes)):
        obs = np.asarray(env.reset(), dtype=np.float32)
        episode_return = 0.0

        # buffer for batched updates
        transition_buffer = []

        for step in range(n_steps):
            a_tanh, logp, entropy_proxy = agent.act(obs, deterministic=False)

            env_action = float(map_action_to_env(a_tanh))

            next_obs, reward = env.step(env_action)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            done = 0#loat(step == n_steps - 1)

            # store transition
            transition_buffer.append(
                (obs, float(reward), next_obs, done, float(a_tanh))
            )

            # performs one batched update every K steps
            if len(transition_buffer) >= update_every or done:
                info = agent.update_batch(transition_buffer)
                if use_wandb:
                    wandb.log(info, step=total_steps)
                transition_buffer.clear()

            total_steps += 1

            obs = next_obs
            episode_return += float(reward)

        episode_returns.append(episode_return)
        ma = float(np.mean(episode_returns[-20:])) if len(episode_returns) >= 20 else float("nan")

        # best moving-average checkpoint
        if ma > best_ma:
            best_ma = ma
            save_actorcritic_checkpoint(
                    agent,
                    episode=episode,
                    swimmer_speed=float(getattr(env, "swimmer_speed", float("nan"))),
                    alignment_timescale=float(getattr(env, "alignment_timescale", float("nan"))),
                    seed=int(getattr(env, "seed", -1) if seed is None else seed),
                    observation_type=str(getattr(env, "observation_type", "unknown")),
                    tag="BEST",
            )


        if logging:
            print(
                f"Episode {episode} return:\t {episode_return:.4f}\t"
                f" actor_loss={info.get('actor_loss', 0.0):.4f}"
                f" critic_loss={info.get('critic_loss', 0.0):.4f}\t MA20={ma:.4f}"
            )

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