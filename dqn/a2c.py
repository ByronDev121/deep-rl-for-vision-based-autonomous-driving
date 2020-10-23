from gym_airsim.airsim_car_env import AirSimCarEnv
import numpy as np
from pathlib import Path
import os
from os.path import exists
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  #dynamically grow the memory used on the GPU
set_session(tf.Session(config=config))

env = AirSimCarEnv()
np.random.seed(123)

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import A2C

log_dir = 'logs'
if not exists(log_dir):
    os.makedirs(log_dir)
checkpoints_dir = Path("logs/a2c_AirsimCar_checkpoints/")
tensorboard_dir = Path("logs/a2c_AirsimCar_tensorboard/")
best_model_dir = Path("logs/a2c_AirsimCar_best_model")
eval_dir = Path("logs/a2c_AirsimCar_eval_results")
if not exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not exists(best_model_dir):
    os.makedirs(best_model_dir)
if not exists(eval_dir):
    os.makedirs(eval_dir)
    

checkpoint_callback = CheckpointCallback(save_freq=25000, save_path=checkpoints_dir,
                                         name_prefix='a2c_AirsimCar')
eval_callback = EvalCallback(env, best_model_save_path=best_model_dir,
                             log_path=eval_dir, eval_freq=10000)

# Create the callback list
callbacks = CallbackList([checkpoint_callback, eval_callback])

model = A2C(CnnPolicy, env, verbose=1, tensorboard_log=tensorboard_dir)
model.learn(total_timesteps=1000000, callback=callbacks, log_interval=10)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

env.close()