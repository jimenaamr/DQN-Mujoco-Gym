# DRL project 2: DRL with value function Q approximation

Link to the used gymnasium environment (walker2d):

https://gymnasium.farama.org/environments/mujoco/walker2d/?utm_source=chatgpt.com


## Launch the training process

1. Start training from zero:

```bash
python -m src.dqn.train --config configs/dqn.yaml
```

2. Resume from a specific checkpoint:

```bash
python -m src.dqn.train \
  --config configs/dqn.yaml \
  --resume checkpoints/dqn/Walker2d-v5_20260218_185009/step_5000000_07-36-34.pt
```

3. Resume from the latest checkpoint of a run **saving videos & checkpoints on the same run directories**:

```bash
python -m src.dqn.train \
  --config configs/dqn.yaml \
  --resume checkpoints/dqn/Walker2d-v5_20260218_185009/
```

4. Resume from the latest checkpoint of a run **saving videos & checkpoints on a new run directories**:

```bash
python -m src.dqn.train \
  --config configs/dqn.yaml \
  --resume checkpoints/dqn/Walker2d-v5_20260218_185009/ \
  --new-run
```

## Open tensorboard webpage

```bash
python -m tensorboard.main --logdir runs --port 6006
```

```bash
python -m src.utils.live_metrics "runs/dqn/Walker2d-v5_20260223_202030"
```


## Avoid GPU crashing

```bash
python -m src.utils.gpu_thermo_guard \
  --gpu 0 \
  --high-c 83 \
  --low-c 77 \
  --power-normal-w 220 \
  --power-throttle-w 160 \
  --poll-s 1.0 \
  --throttle-min-s 30
```