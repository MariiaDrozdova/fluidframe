import argparse
import numpy as np
import torch
from typing import Dict

from environments.taylor_green import TaylorGreenEnvironment
from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment

from eval import plot_policy

from agents.agent_saclearning import SACAgent, SACConfig, map_action_to_env
from agents.agent_AClearning import ActorCriticAgent, ActorCriticConfig

# -------------------------------------------------
# loaders
# -------------------------------------------------

def load_sac_agent(path: str, obs_dim: int):
    ckpt = torch.load(path, map_location="cpu")
    cfg = SACConfig(obs_dim=obs_dim, act_dim=1)
    agent = SACAgent(cfg)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor.eval()
    return agent

def load_AC_agent(path: str, obs_dim: int):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ActorCriticConfig(obs_dim=obs_dim, act_dim=1)
    agent = ActorCriticAgent(cfg)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor.eval()
    return agent


def load_agent(algo: str, path: str, obs_dim: int):
    if algo == "SAC":
        return load_sac_agent(path, obs_dim)
    if algo == "AC":
        return load_AC_agent(path, obs_dim)
    raise ValueError("Unknown algo")


# -------------------------------------------------
# evaluation
# -------------------------------------------------

def eval_checkpoint(
    checkpoint: str,
    algo: str,
    swimmer_speed: float,
    alignment_timescale: float,
    n_episodes: int,
    n_steps: int,
    deterministic: bool = True,
    make_plot: bool = True,
):

    rng = np.random.default_rng(42)

    env = TaylorGreenContinuousEnvironment(
        dt=0.01,
        swimmer_speed=swimmer_speed,
        alignment_timescale=alignment_timescale,
        seed=42,
    )

    env_naive = TaylorGreenEnvironment(
        dt=0.01,
        swimmer_speed=swimmer_speed,
        alignment_timescale=alignment_timescale,
        seed=42,
    )

    obs0 = np.asarray(env.reset(), dtype=np.float32)
    agent = load_agent(algo, checkpoint, obs_dim=len(obs0))

    total_return = 0.0
    total_return_naive = 0.0

    positions = np.zeros((n_steps, 2, n_episodes))
    positions_naive = np.zeros((n_steps, 2, n_episodes))

    for ep in range(n_episodes):

        pos0 = np.array([rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)])
        theta0 = rng.uniform(0, 2*np.pi)

        obs = env.reset(pos0.copy(), theta0)
        env_naive.reset(pos0.copy(), theta0)

        ep_ret = 0.0
        ep_ret_naive = 0.0

        for t in range(n_steps):

            if algo == "AC":
                a_tanh, _, _ = agent.act(obs, deterministic=deterministic)
            else:
                a_tanh = agent.act(obs, deterministic=deterministic)

            action = map_action_to_env(a_tanh)

            obs, r = env.step(action)
            _, r_naive = env_naive.step(1)

            ep_ret += r
            ep_ret_naive += r_naive

            positions[t, :, ep] = env.swimmer_position
            positions_naive[t, :, ep] = env_naive.swimmer_position

        total_return += ep_ret
        total_return_naive += ep_ret_naive

        print(f"Episode {ep+1} return {ep_ret:.3f}  naive {ep_ret_naive:.3f}")

    print()
    print("Mean return:", total_return / n_episodes)
    print("Mean naive :", total_return_naive / n_episodes)

    if total_return_naive != 0:
        print("Gain:", total_return / total_return_naive - 1)

    if make_plot:
        plot_params = dict(phi=swimmer_speed, psi=alignment_timescale, algo=algo)
        plot_policy(n_episodes, positions, positions_naive, plot_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--algo", choices=["SAC","AC",], default="SAC")
    parser.add_argument("--swimmer-speed", type=float, default=0.3)
    parser.add_argument("--alignment-timescale", type=float, default=1.0)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--make-plot", type=bool, default=True)
    args = parser.parse_args()

    eval_checkpoint(
        checkpoint=args.checkpoint,
        algo=args.algo,
        swimmer_speed=args.swimmer_speed,
        alignment_timescale=args.alignment_timescale,
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        deterministic=args.deterministic,
        make_plot=args.make_plot,
    )