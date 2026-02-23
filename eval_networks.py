import argparse
import numpy as np
import torch
try:
    import wandb
except Exception:  # wandb is optional
    wandb = None
from typing import Dict

from environments.taylor_green import TaylorGreenEnvironment
from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment

from eval import plot_policy

from agents.agent_saclearning import SACAgent, SACConfig, map_action_to_env
from agents.agent_AClearning import ActorCriticAgent, ActorCriticConfig
from agents.agent_mpolearning import MPOAgent, MPOConfig

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

def load_mpo_agent(path: str, obs_dim: int):
    ckpt = torch.load(path, map_location="cpu")
    cfg = MPOConfig(obs_dim=obs_dim, act_dim=1)
    agent = MPOAgent(cfg)
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
    elif algo == "AC":
        return load_AC_agent(path, obs_dim)
    elif algo == "MPO":
        return load_mpo_agent(path, obs_dim)
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
    observation_type: str = "velocity",
    log_wandb: bool = False,
    wandb_project: str = "fluidframe-eval",
    reward_scale: str = "raw",
):

    rng = np.random.default_rng(42)

    if log_wandb and wandb is None:
        raise ImportError("wandb is not installed but --log-wandb was set")

    env = TaylorGreenContinuousEnvironment(
        dt=0.01,
        swimmer_speed=swimmer_speed,
        alignment_timescale=alignment_timescale,
        seed=42,
        action_type="continuous",
        observation_type=observation_type,
    )

    env_naive = TaylorGreenEnvironment(
        dt=0.01,
        swimmer_speed=swimmer_speed,
        alignment_timescale=alignment_timescale,
        seed=42,
    )

    obs0 = np.asarray(env.reset(), dtype=np.float32)
    agent = load_agent(algo, checkpoint, obs_dim=len(obs0))

    if log_wandb:
        run_name = f"{algo}-phi{swimmer_speed}-psi{alignment_timescale}-obs{observation_type}-det{int(deterministic)}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "algo": algo,
                "checkpoint": checkpoint,
                "swimmer_speed": swimmer_speed,
                "alignment_timescale": alignment_timescale,
                "dt": float(getattr(env, "dt", 0.01)),
                "n_episodes": n_episodes,
                "n_steps": n_steps,
                "deterministic": deterministic,
                "make_plot": make_plot,
                "observation_type": observation_type,
                "reward_scale": reward_scale,
            },
        )

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
            r = float(r)
            _, r_naive = env_naive.step(1)
            r_naive = float(r_naive)

            ep_ret += r
            ep_ret_naive += r_naive

            positions[t, :, ep] = env.swimmer_position
            positions_naive[t, :, ep] = env_naive.swimmer_position

        total_return += ep_ret
        total_return_naive += ep_ret_naive

        # final position stats
        final_x, final_y = float(env.swimmer_position[0]), float(env.swimmer_position[1])
        final_x_naive, final_y_naive = float(env_naive.swimmer_position[0]), float(env_naive.swimmer_position[1])

        print(f"Episode {ep+1} return {ep_ret:.3f}  naive {ep_ret_naive:.3f}")

        if log_wandb:
            wandb.log(
                {
                    "eval/episode_return": float(ep_ret),
                    "eval/episode_return_naive": float(ep_ret_naive),
                    "eval/episode_gain": float(ep_ret / ep_ret_naive - 1.0) if ep_ret_naive != 0 else 0.0,
                    "eval/final_x": final_x,
                    "eval/final_y": final_y,
                    "eval/final_x_naive": final_x_naive,
                    "eval/final_y_naive": final_y_naive,
                },
                step=ep,
            )

    print()
    print("Mean return:", total_return / n_episodes)
    print("Mean naive :", total_return_naive / n_episodes)

    mean_ret = total_return / n_episodes
    mean_naive = total_return_naive / n_episodes
    gain = (mean_ret / mean_naive - 1.0) if mean_naive != 0 else float("nan")

    if total_return_naive != 0:
        print("Gain:", total_return / total_return_naive - 1)

    if log_wandb:
        wandb.summary["eval/return_mean"] = float(mean_ret)
        wandb.summary["eval/naive_return_mean"] = float(mean_naive)
        wandb.summary["eval/gain_vs_naive"] = float(gain)

    if make_plot:
        plot_params = dict(phi=swimmer_speed, psi=alignment_timescale, algo=algo)
        plot_policy(n_episodes, positions, positions_naive, plot_params)

    if log_wandb:
        # Try to upload the generated PDF plot, if present
        try:
            plot_path = f"{algo}_phi{swimmer_speed}_psi{alignment_timescale}.pdf"
            wandb.save(plot_path)
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--algo", choices=["SAC","AC", "MPO"], default="SAC")
    parser.add_argument("--swimmer-speed", type=float, default=0.3)
    parser.add_argument("--alignment-timescale", type=float, default=1.0)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--make-plot", type=bool, default=True)
    parser.add_argument("--observation-type", type=str, choices=["original", "velocity"], default="velocity")
    parser.add_argument("--log-wandb", action="store_true", help="Log eval metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="fluidframe-eval")
    parser.add_argument("--reward-scale", type=str, choices=["raw", "per_dt"], default="raw")
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
        observation_type=args.observation_type,
        log_wandb=args.log_wandb,
        wandb_project=args.wandb_project,
        reward_scale=args.reward_scale,
    )