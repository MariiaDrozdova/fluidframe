#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1

N_STEPS=5000
EP_SAC=150
EP_MPO=150
EP_AC=1000

# (swimmer_speed, alignment_timescale) triplets
PHIS=(0.1 1.0 0.3)
PSIS=(0.3 1.0 1.0)

ALGOS=("SAC" "MPO")

for i in "${!PHIS[@]}"; do
  PHI="${PHIS[$i]}"
  PSI="${PSIS[$i]}"

  echo "=============================="
  echo "Running sweep case: phi=${PHI}, psi=${PSI}"
  echo "=============================="

  for ALGO in "${ALGOS[@]}"; do
    if [[ "${ALGO}" == "AC" ]]; then
      EP="${EP_AC}"
    elif [[ "${ALGO}" == "SAC" ]]; then
      EP="${EP_SAC}"
    else
      EP="${EP_MPO}"
    fi

    echo ""
    echo "--- ${ALGO} | phi=${PHI} psi=${PSI} | episodes=${EP} steps=${N_STEPS} ---"
    python main.py \
      --swimmer-speed "${PHI}" \
      --alignment-timescale "${PSI}" \
      --n-episodes "${EP}" \
      --n-steps "${N_STEPS}" \
      --train-type "${ALGO}" \
      --use-continuous-environment \
      --use-wandb
  done
done
