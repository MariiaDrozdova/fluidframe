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

