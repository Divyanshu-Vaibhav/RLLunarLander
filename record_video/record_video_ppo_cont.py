import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC, PPO

def record_policy(model_path, env_id, out_dir, episodes=3, deterministic=True):
    env = gym.make(env_id, render_mode="rgb_array") 
    env = RecordVideo(env, video_folder=out_dir, episode_trigger=lambda eid: True, name_prefix="eval")
    model = (SAC if "sac" in model_path.lower() else PPO).load(model_path)
    for ep in range(episodes):
        obs, info = env.reset(seed=100+ep)
        done = False; trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, trunc, info = env.step(action)
    env.close()

record_policy("./models/ppo_lander_cont_best/best_model", "LunarLanderContinuous-v3", "./videos/ppo_cont")
