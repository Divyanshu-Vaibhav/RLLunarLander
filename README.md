# Lunar Lander

Reproducible RL benchmark comparing PPO, DQN (discrete) and PPO, SAC (continuous) on Gymnasium LunarLander v3 environments using Stable-Baselines3, with seeded evaluation, TensorBoard logs, checkpoints, and demo videos.

## 🚀 Quick Overview

- **Algorithms**: PPO, DQN, SAC with standardized evaluation protocol
- **Environments**: LunarLander-v3 (discrete), LunarLanderContinuous-v3 (continuous)
- **Framework**: Stable-Baselines3 with Gymnasium v3
- **Evaluation**: Fixed seeds, 10-episode metrics, TensorBoard logging

## 📊 Results Summary

| Algorithm | Environment | Mean Reward | Std Dev | Episode Length | Training Steps |
|-----------|-------------|-------------|---------|----------------|----------------|
| PPO | Discrete | 217 | ±15 | ~437 | 1,000,000 |
| DQN | Discrete | 121 | ±89 | ~588 | 1,500,000 |
| PPO | Continuous | TBD | TBD | TBD | 1,500,000 |
| SAC | Continuous | 282 | TBD | ~201 | 750,000 |

## 🎥 Demo Videos


### PPO (Discrete Control)
<video src="https://github.com/user-attachments/assets/b8a1f3e8-c22b-414a-aab5-558bda2715ff
" controls width="600"></video>

### DQN (Discrete Control)  
<video src="https://github.com/user-attachments/assets/b64334f9-b371-40a4-bfd8-7efbc895cc2c
" controls width="600"></video>

### PPO (Continuous Control)
<video src="https://github.com/user-attachments/assets/d4229c00-8b58-407a-9c72-12194813df34
" controls width="600"></video>

### SAC (Continuous Control)
<video src="https://github.com/user-attachments/assets/b883f720-fb0b-4380-8b68-1e6e3ebc2a2a
" controls width="600"></video>

