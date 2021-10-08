# from airsim_gym.gym import AirSimCarEnv
# import numpy as np
# from pathlib import Path
import os
from os.path import exists
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# set_session(tf.Session(config=config))

# env = AirSimCarEnv()
# np.random.seed(123)

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import A2C

# log_dir = 'logs_a2c'
# if not exists(log_dir):
#     os.makedirs(log_dir)
# checkpoints_dir = Path("logs_a2c/a2c_AirsimCar_checkpoints/")
# tensorboard_dir = Path("logs_a2c/a2c_AirsimCar_tensorboard/")
# best_model_dir = Path("logs_a2c/a2c_AirsimCar_best_model")
# eval_dir = Path("logs_a2c/a2c_AirsimCar_eval_results")
# if not exists(checkpoints_dir):
#     os.makedirs(checkpoints_dir)
# if not exists(tensorboard_dir):
#     os.makedirs(tensorboard_dir)
# if not exists(best_model_dir):
#     os.makedirs(best_model_dir)
# if not exists(eval_dir):
#     os.makedirs(eval_dir)

class A2C_:

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
            name_prefix='a2c_AirsimCar'
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=self.best_model_dir,
            log_path=self.eval_dir,
            eval_freq=10000
        )

        # Create the callback list
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        model = A2C(CnnPolicy, env, verbose=1, tensorboard_log=self.tensorboard_dir)
        model.learn(total_timesteps=1000000, callback=callbacks, log_interval=1)

        # obs = env.reset()
        # for i in range(1000):
        #     action, _states = model.predict(obs)
        #     obs, rewards, done, info = env.step(action)
        #
        # env.close()