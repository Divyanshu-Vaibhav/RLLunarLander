import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

ENV_ID = "LunarLander-v3"  # Gymnasium v3 IDs
SEED = 42
TOTAL_STEPS = 1_000_000
LOG_DIR = "./runs/ppo_lander_v3"
BEST_DIR = "./models/ppo_lander_v3_best"
CKPT_DIR = "./models/ppo_lander_v3_ckpt"

os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# The PPO was intended to run on CPU
DEVICE = "cpu"

def make_env(seed: int):
    def _thunk():
        env = gym.make(ENV_ID)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk

if __name__ == "__main__":
    # Reproducibility
    set_random_seed(SEED)

    # Training env (vectorized for PPO advantage estimation)
    train_env = DummyVecEnv([make_env(SEED)])

    # Separate eval env
    eval_env = DummyVecEnv([make_env(SEED + 100)])

    # Callbacks: evaluate every 10k steps, save best, checkpoint periodically, optional stop at target reward
    reward_threshold = 250.0  # optional early stop if consistently solved
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb
    )
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=CKPT_DIR, name_prefix="ppo_lander")

    # PPO hyperparameters: 
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        learning_rate=2e-4,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device=DEVICE,
        seed=SEED,
    )


    model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb], tb_log_name="ppo_lander_run")

    # Save final model
    model.save(os.path.join(CKPT_DIR, "ppo_lander_final"))
    train_env.close()
    eval_env.close()

    print("Training complete. Launch TensorBoard with:")
    print("  tensorboard --logdir ./runs")
    