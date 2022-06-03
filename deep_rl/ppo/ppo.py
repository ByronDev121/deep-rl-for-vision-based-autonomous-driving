import os
from os.path import exists
from stable_baselines.common.policies import CnnPolicy
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

        self.model = None

    def load_weights(self, model_path):
        self.model = PPO2.load(model_path)

    def policy_action(self, state):
        return self.model.predict(state)

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

        model.learn(total_timesteps=1000000, callback=callbacks, log_interval=1)

        return
