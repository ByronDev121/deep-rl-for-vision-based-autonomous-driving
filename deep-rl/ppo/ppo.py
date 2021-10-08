# from gym_airsim.airsim_car_env import AirSimCarEnv
import numpy as np
from pathlib import Path
import os
from os.path import exists

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# set_session(tf.Session(config=config))

# env = AirSimCarEnv()
# np.random.seed(123)

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList


class PPO_2:

    def __init__(self, save_dir):
        self.checkpoints_dir = '{}/checkpoints'.format(save_dir)
        self.tensorboard_dir = '{}/tensorboard'.format(save_dir)
        self.best_model_dir = '{}/best_model'.format(save_dir)
        self.eval_dir = '{}/eval'.format(save_dir)
        if not exists(self.eval_dir):
            os.makedirs(self.eval_dir)

    def train(self, env, summary_writer):
        checkpoint_callback = CheckpointCallback(
            save_freq=25000,
            save_path=self.checkpoints_dir,
            # name_prefix='ppo2_AirsimCar'
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=self.best_model_dir,
            log_path=self.eval_dir,
            eval_freq=25000
        )

        # Create the callback list
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=self.tensorboard_dir)

        # Add some param noise for exploration
        # param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)

        # Because we use parameter noise, we should use a MlpPolicy with layer normalization
        # model = PPO2(LnMlpPolicy, env, verbose=1, tensorboard_log=tensorboard_dir)

        model.x(total_timesteps=1000000, callback=callbacks, log_interval=1)

        # obs = env.reset()
        # for i in range(1000):
        #     action, _states = model.predict(obs)
        #     obs, rewards, done, info = env.step(action)
        #
        # env.close()

        return
