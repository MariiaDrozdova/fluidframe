import argparse

from environments.taylor_green import TaylorGreenEnvironment
from environments.taylor_green_dedalus import TaylorGreenDedalusEnvironment
from environments.taylor_green_continuous import TaylorGreenContinuousEnvironment
from train import train
from trainers.train_ac import train_actorcritic
from trainers.train_sac import train_sac
from trainers.train_mpo import train_mpo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swimmer-speed", type=float, default=0.3)
    parser.add_argument("--alignment-timescale", type=float, default=1.0)
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--n-steps", type=int, default=100000)
    parser.add_argument("--use-dedalus-environment", action="store_true", default=False)
    parser.add_argument("--use-continuous-environment", action="store_true", default=False)
    parser.add_argument("--train-type", type=str, choices=["qlearning", "MPO", "SAC", "AC"], default="qlearning")
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--observation-type", type=str, choices=["original", "velocity"], default="velocity")
    parser.add_argument("--c", type=float, default=10.0, help="reward scaling factor for SAC and MPO")
    args = parser.parse_args()

    if args.use_continuous_environment:
        print("Using continuous environment with continuous actions ...")
        env = TaylorGreenContinuousEnvironment(
            dt=0.01,
            swimmer_speed=args.swimmer_speed,
            alignment_timescale=args.alignment_timescale,
            seed=42,
            action_type="continuous",
            observation_type=args.observation_type,
        )
    elif args.use_dedalus_environment:
        print("Using dedalus to specify flow variables ...")
        env = TaylorGreenDedalusEnvironment(
            dt=0.01,
            swimmer_speed=args.swimmer_speed,
            alignment_timescale=args.alignment_timescale,
            seed=42,
        )  # initialise environment
    else:
        print("Using closed-form analytical solution ...")
        env = TaylorGreenEnvironment(
            dt=0.01,
            swimmer_speed=args.swimmer_speed,
            alignment_timescale=args.alignment_timescale,
            seed=42,
        )  # initialise environment

    if args.train_type == "SAC":
        print("Training SAC agent ...")
        train_sac(
            env=env,
            n_episodes=args.n_episodes,
            n_steps=args.n_steps,
            save=True,
            seed=42,
            use_wandb=args.use_wandb,
            c=args.c,
        )
    elif args.train_type == "MPO":
        print("Training MPO agent ...")
        train_mpo(
            env=env,
            n_episodes=args.n_episodes,
            n_steps=args.n_steps,
            save=True,
            seed=42,
            use_wandb=args.use_wandb,
            c=args.c,
        )
    elif args.train_type == "AC":
        print("Training Actor-Critic agent ...")
        train_actorcritic(
            env=env,
            n_episodes=args.n_episodes,
            n_steps=args.n_steps,
            save=True,
            seed=42,
        )
    elif args.train_type == "qlearning":
        print("Training Q-learning agent ...")
        train(
            env=env,
            n_episodes=args.n_episodes,
            n_steps=args.n_steps,
            save=True,
            seed=42,
        )

