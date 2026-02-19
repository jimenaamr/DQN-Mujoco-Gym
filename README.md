# DRL project 2: DRL with value function Q approximation

Link to the used gymnasium environment (walker2d):

https://gymnasium.farama.org/environments/mujoco/walker2d/?utm_source=chatgpt.com


## How you control the checkpoint (what gets loaded and which one is chosen)

1. “I want to resume from a specific checkpoint”
   Pass the exact file:

```bash
python -m src.DQN_walker2d.train --config configs/dqn.yaml \
  --resume checkpoints/Walker2d-v5_20260218_185009/step_5000000_07-36-34.pt
```

This avoids any heuristics.

2. “I want to resume from the latest checkpoint of a run”
   Pass the directory:

```bash
python -m src.DQN_walker2d.train --config configs/dqn.yaml \
  --resume checkpoints/Walker2d-v5_20260218_185009/
```
