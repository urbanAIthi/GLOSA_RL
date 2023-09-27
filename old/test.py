import time
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb
from GLOSA_gym.environment import SumoGlosaEnv


config = {"policy_type": "MlpPolicy", "total_timesteps": 25000}
experiment_name = f"PPO_{int(time.time())}"
wandb.init(
    name=experiment_name,
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

def make_env():
    env = SumoGlosaEnv()
    env = Monitor(env)  # record stats such as returns
    return env
env = DummyVecEnv([make_env])
#env = VecVideoRecorder(env, "videos",
#    record_video_trigger=lambda x: x % 2000 == 0, video_length=200)  # record videos
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{experiment_name}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_freq=1000,
        model_save_path=f"models/{experiment_name}",
    ),
)