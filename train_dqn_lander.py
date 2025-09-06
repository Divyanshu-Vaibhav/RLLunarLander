import os
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

ENV_ID = "LunarLander-v3"
SEED = 42
TOTAL_STEPS = 1_500_000  # DQN often needs more steps than PPO
LOG_DIR = "./runs/dqn_lander_v3"
BEST_DIR = "./models/dqn_lander_v3_best"
CKPT_DIR = "./models/dqn_lander_v3_ckpt"

os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# RL tasks perform better on CPU
DEVICE = "cpu"


def make_env(seed: int):
    def _thunk():
        env = gym.make(ENV_ID)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk

if __name__ == "__main__":
    set_random_seed(SEED)

    # DQN is off-policy and single-env; keep DummyVecEnv for Monitor/eval compatibility
    train_env = DummyVecEnv([make_env(SEED)])
    eval_env = DummyVecEnv([make_env(SEED + 100)])

    # A solid DQN config for LunarLander (inspired by RL Zoo and docs)
    policy_kwargs = dict(net_arch=[256, 256])

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=6.3e-4,         # try 1e-4 to 6.3e-4 in sweeps
        buffer_size=200_000,          # replay buffer
        learning_starts=10_000,       # warmup before learning
        batch_size=128,
        gamma=0.99,
        train_freq=4,                 # env steps per gradient step
        gradient_steps=1,             # 1 gradient step per train_freq
        target_update_interval=1000,  # hard update frequency
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.12,    # decay 12% of training
        max_grad_norm=10,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device=DEVICE,
        seed=SEED,
        policy_kwargs=policy_kwargs,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=CKPT_DIR, name_prefix="dqn_lander")

    model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb], tb_log_name="dqn_lander_run")

    model.save(os.path.join(CKPT_DIR, "dqn_lander_final"))
    train_env.close()
    eval_env.close()

    print("Training complete. Launch TensorBoard with:")
    print("  tensorboard --logdir ./runs")
