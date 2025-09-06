import os, gymnasium as gym, torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

ENV_ID = "LunarLanderContinuous-v3"
SEED = 42
TOTAL_STEPS = 1_500_000
LOG_DIR = "./runs/ppo_lander_cont_v3"
BEST_DIR = "./models/ppo_lander_cont_best"
CKPT_DIR = "./models/ppo_lander_cont_ckpt"
os.makedirs(BEST_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cpu"  
def make_env(seed):
    def _thunk():
        env = gym.make(ENV_ID)
        env = Monitor(env); env.reset(seed=seed); return env
    return _thunk

set_random_seed(SEED)
train_env = DummyVecEnv([make_env(SEED)])
eval_env = DummyVecEnv([make_env(SEED+100)])
eval_cb = EvalCallback(eval_env, best_model_save_path=BEST_DIR, log_path=LOG_DIR,
                       eval_freq=10_000, n_eval_episodes=10, deterministic=True)
ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=CKPT_DIR, name_prefix="ppo_lander_cont")
model = PPO("MlpPolicy", train_env,
            n_steps=4096, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
            tensorboard_log=LOG_DIR, verbose=1, device=DEVICE, seed=SEED)

model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb], tb_log_name="ppo_cont_run")
model.save(os.path.join(CKPT_DIR, "ppo_cont_final"))
train_env.close(); eval_env.close()
