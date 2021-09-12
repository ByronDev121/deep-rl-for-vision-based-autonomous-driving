from gym_airsim.airsim_car_env import AirSimCarEnv
import numpy as np
from pathlib import Path
import os
from os.path import exists

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
set_session(tf.Session(config=config))

env = AirSimCarEnv()
np.random.seed(123)

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList


log_dir = 'logs_ppo'

if not exists(log_dir):
    os.makedirs(log_dir)
checkpoints_dir = Path("logs_ppo/ppo2_AirsimCar_checkpoints/")
tensorboard_dir = Path("logs_ppo/ppo2_AirsimCar_tensorboard/")
best_model_dir = Path("logs_ppo/ppo2_AirsimCar_best_model")
eval_dir = Path("logs_ppo/ppo2_AirsimCar_eval_results")
if not exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not exists(best_model_dir):
    os.makedirs(best_model_dir)
if not exists(eval_dir):
    os.makedirs(eval_dir)

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path=checkpoints_dir,
    name_prefix='ppo2_AirsimCar'
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=best_model_dir,
    log_path=eval_dir,
    eval_freq=25000
)

# Create the callback list
callbacks = CallbackList([checkpoint_callback, eval_callback])

# model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=tensorboard_dir)

# Add some param noise for exploration
# param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)

# Because we use parameter noise, we should use a MlpPolicy with layer normalization
model = PPO2(LnMlpPolicy, env, verbose=1, tensorboard_log=tensorboard_dir)


model.learn(total_timesteps=1000000, callback=callbacks, log_interval=1)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#
# env.close()
