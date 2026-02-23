# Neural Policy Experiments (Continuous State)

This part of the project explores learning control policies directly from continuous observations of the swimmer in the Taylor-Green flow.

What was implemented:
- Continuous observation space instead of the original discrete grid.
- Neural policies trained with Actor-Critic (AC) and Soft Actor-Critic (SAC).

Observation state:
- swimmer orientation (encoded continuously)
- local flow velocity, normalized by the flow scale
- local vorticity, normalized by the flow scale

A minimal state using only orientation and vorticity was briefly tested. In a small number of quick runs it appeared unstable and inconsistent, so (for now) we switched to a richer local-flow observation.

### Main findings (preliminary)
- Actor-Critic learns stable and strong policies in the continuous environment, but required batching experience (a small buffer) to stabilize updates.
- Soft Actor-Critic is much more sensitive to reward magnitude.
- Performance varies strongly with physical parameters (e.g. alignment timescale).

These observations come from fast exploratory experiments and should be considered preliminary.

# Running the code

Below are example commands used for training and evaluating the neural policies.

### Train Actor-Critic (continuous environment)
```
OMP_NUM_THREADS=1 python main.py \
    --swimmer-speed 0.4 \
    --alignment-timescale 1.0 \
    --n-episodes 1000 \
    --n-steps 5000 \
    --train-type AC \
    --use-continuous-environment
```

### Evaluate Actor-Critic checkpoint
```
python eval_networks.py \
    --checkpoint ./checkpoints/ac_ep999.pt \
    --algo AC \
    --swimmer-speed 0.4 \
    --alignment-timescale 1.0
```

### Train Soft Actor-Critic (continuous environment)
```
OMP_NUM_THREADS=1 python main.py \
    --swimmer-speed 0.4 \
    --alignment-timescale 1.0 \
    --n-episodes 100 \
    --n-steps 5000 \
    --train-type sac \
    --use-continuous-environment
```

### Evaluate SAC checkpoint
```
python eval_networks.py \
    --checkpoint ./checkpoints/sac_ep99.pt \
    --algo SAC \
    --swimmer-speed 0.4 \
    --alignment-timescale 1.0
```

# SAC 
```
for each update:

    batch ← replay buffer

    # ----- critic -----
    a_next ~ π(s_next)
    target = r + γ (Q_target(s_next, a_next) - α logπ(a_next))
    update Q networks toward target

    # ----- actor -----
    a ~ π(s)
    actor_loss = α logπ(a) - Q(s,a)
    update actor

    # ----- target networks -----
    soft update Q_target
```

# MPO

```
for each update:

    batch ← replay buffer

    # ----- critic -----
    compute targets (often Retrace or TD(0)/TD(λ))
    update Q networks

    # ----- E-step (per state) -----
    for each state s in batch:
        sample many actions a_i ~ π_old(·|s)
        compute Q(s, a_i)

        # 1) solve for η (temperature) to satisfy KL constraint
        find η* that makes:
            KL( q(·|s) || π_old(·|s) ) ≈ ε

        # 2) compute nonparametric q weights using η*
        w_i ∝ exp(Q(s,a_i) / η*)
        normalize w_i to sum to 1

    # ----- M-step -----
    update policy parameters θ to maximize:
        Σ_s Σ_i w_i log π_θ(a_i | s)
    subject to:
        KL( π_old(·|s) || π_θ(·|s) ) ≤ ε_m
```

