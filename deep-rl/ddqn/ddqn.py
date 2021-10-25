import random
import numpy as np
import time
import tensorflow as tf
from os.path import dirname, abspath, join
from tqdm import tqdm
from threading import Thread
from collections import deque
from random import random, randrange
from configparser import ConfigParser

from .agent import Agent
from .memory_buffer import MemoryBuffer
from Masters.utils.image_processing import ImageProcessing


class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, args, save_dir):
        """ Initialization
        """
        self.save_dir = save_dir
        # Environment and ddqn parameters
        self.model_type = args.model_type
        self.double_deep = args.double_deep
        self.with_hrs = args.with_hrs
        self.augment = args.augment
        #

        config = ConfigParser()
        config.read(join(dirname(abspath(__file__)), '..', '..', 'airsim_gym', 'config.ini'))

        state_height = int(config['car_agent']['state_height'])
        state_width = int(config['car_agent']['state_width'])
        consecutive_frames = int(config['car_agent']['consecutive_frames'])
        act_dim = int(config['car_agent']['act_dim'])
        max_steering_angle = float(config['car_agent']['max_steering_angle'])
        self.fps = int(config['car_agent']['fps'])
        self.state_dim = (state_height, state_width, consecutive_frames)
        self.act_dim = act_dim
        #
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_final = args.epsilon_final
        self.replay_buffer_size = args.replay_buffer_size
        self.replay_start_size = args.replay_start_size
        self.batch_size = args.batch_size
        self.tau = 1.0
        #
        self.nb_episodes = args.nb_episodes
        self.train_in_loop = args.train_in_loop
        self.nb_steps_per_train_iter = args.nb_steps_per_train_iter
        self.target_network_update = args.target_network_update
        #
        # Create q-network
        self.agent = Agent(self.model_type, self.state_dim, self.act_dim, self.lr, self.tau, args.dueling)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.replay_buffer_size)
        #
        self.processing = ImageProcessing(self.state_dim[1], self.state_dim[2], self.state_dim[0], act_dim,
                                          max_steering_angle)
        #
        self.average_cumul_reward = deque()
        self.best_episode_reward = 0
        self.training_initialized = False
        self.terminate = False
        self.avg_fps = 0

    @staticmethod
    def tf_summary(tag, val):
        """ Scalar Value Tensorflow Summary
        """
        return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])

    def train(self, env, summary_writer):
        """ Main ddqn Training Algorithm
        """

        results = []
        total_steps, train_count = 0, 0
        tqdm_e = tqdm(range(self.nb_episodes), desc='Score', leave=True, unit=" episodes")

        if self.train_in_loop:
            # Start training thread and wait for training to be initialized
            trainer_thread = Thread(target=self._train_in_loop, daemon=True)
            trainer_thread.start()
            while not self.training_initialized:
                time.sleep(0.01)

        for e in tqdm_e:
            # Reset episode
            episode_steps, cumul_reward, episode_qs, done = 0, 0, 0, False
            state = env.reset()

            start_time = time.time()

            while not done:
                step_start_time = time.time()
                # Actor picks an action (following the epsilon-greedy policy)
                a, q = self.policy_action(state)
                episode_qs += q
                # Retrieve new state, reward, and whether the state is a terminal state
                new_state, r, done, _ = env.step(a)
                #  augment data
                if self.augment:
                    state, new_state, a = self.processing.augment(state, new_state, a)
                # Memorize for experience replay
                self._memorize(state, a, r, done, new_state)
                # Update current state
                state = new_state
                cumul_reward += r
                episode_steps += 1
                total_steps += 1

                if not self.train_in_loop:
                    # Train agent
                    if self.buffer.size() > self.replay_start_size \
                            and total_steps % self.nb_steps_per_train_iter == 0:
                        self._train_agent()
                        train_count += 1
                        if train_count % self.target_network_update == 0:
                            self.agent.transfer_weights()
                        time.sleep(0.01)

                step_time = time.time() - step_start_time
                if step_time < (1 / self.fps):
                    time.sleep((1 / self.fps) - step_time)

            if cumul_reward > self.best_episode_reward:
                self.best_episode_reward = cumul_reward
                self.save_weights('{}/best_model/best_model'.format(self.save_dir))

            # Export results for Tensorboard
            score = self.tf_summary('reward', cumul_reward)
            avg_score = self.tf_summary('average-reward-per-step', cumul_reward / episode_steps)
            qs = self.tf_summary('q-value', episode_qs)
            avg_qs = self.tf_summary('average-q-per-step', episode_qs / episode_steps)
            eps = self.tf_summary('epsilon', self.epsilon)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.add_summary(avg_score, global_step=e)
            summary_writer.add_summary(qs, global_step=e)
            summary_writer.add_summary(avg_qs, global_step=e)
            summary_writer.add_summary(eps, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()
            time.sleep(0.01)

            # Display stats
            self.avg_fps = episode_steps / (time.time() - start_time)
            print('\n')
            print('Average Reward: {}'.format(cumul_reward / episode_steps))
            print('Average Q: {}'.format(episode_qs / episode_steps))
            print('Epsilon: {}'.format(self.epsilon))
            print('Buffer Size: {}'.format(self.buffer.size()))
            print('Average FPS: {}'.format(self.avg_fps))
            print('Total Steps: {}'.format(total_steps))
            print('\n')

            if e % 100 == 0:
                self.save_weights('{}/checkpoints/{}_{}'.format(
                    self.save_dir,
                    'dqn',
                    e
                ))

            # Decay epsilon
            if self.epsilon > self.epsilon_final:
                self.epsilon *= self.epsilon_decay

        if self.train_in_loop:
            trainer_thread.join()

        return results

    def save_weights(self, path):
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.act_dim), 0

        else:
            pred = self.agent.predict(s)[0]
            return np.argmax(pred), max(pred)

    def _memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def _train_in_loop(self):
        # first fit is slow:
        x = np.random.uniform(size=self.state_dim).astype(np.float32)
        y = np.random.uniform(size=(1, self.act_dim)).astype(np.float32)
        self.agent.fit(x, y)
        self.training_initialized = True
        # Train in loop
        count = 0
        total_iter_time = 0
        while True:
            if self.terminate:
                return
            elif self.buffer.size() >= self.replay_start_size:
                start_time = time.time()
                self._train_agent()
                if count % self.target_network_update == 0:
                    print('Transferring Weights to Target Network')
                    self.agent.transfer_weights()
                count += 1
                total_iter_time += time.time() - start_time
                time.sleep(self._get_sleep_time(count, total_iter_time))
            else:
                time.sleep(10)

    def _get_sleep_time(self, count, total_iter_time):
        average_iter_time = total_iter_time / count
        sleep_time = (1 / (self.avg_fps / self.nb_steps_per_train_iter)) - average_iter_time
        if sleep_time < 0:
            sleep_time = 0.01
        return sleep_time

    def _train_agent(self):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer
        s, a, r, d, new_s, idx = self.buffer.sample_batch(self.batch_size)

        # Apply Bellman Equation on batch samples to train our ddqn
        q = self.agent.predict(s)
        q_targ = self.agent.target_predict(new_s)

        if self.double_deep:
            next_q = self.agent.predict(new_s)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                if self.double_deep:
                    # print("double deep q networks")
                    best_next_action = np.argmax(next_q[i, :])
                    q[i, a[i]] = r[i] + self.gamma * q_targ[i, best_next_action]
                else:
                    # print("deep q networks")
                    q[i, a[i]] = r[i] + self.gamma * np.max(q_targ[i])


        # Train on batch
        self.agent.fit(s, q)
